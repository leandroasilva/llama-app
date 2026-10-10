import Foundation
import os.log

/// Turns a `llama://install?repo=…` deeplink into a concrete file set:
/// a specific GGUF (possibly sharded) and an optional mmproj sidecar, all
/// expressed as HF resolve URLs we can hand to `ModelManager`.
///
/// File-selection priority:
///   1. Explicit `quant` → sibling whose `parseGGUFQuantLabel` matches.
///   2. No quant → mirror llama.cpp's `find_best_model` (try Q4_K_M tag,
///      then Q4_0, else fall back to largest fits-budget). Pre-filters
///      candidates by `memBudget` so a small Mac doesn't get handed a
///      too-big Q4_K_M; final fallback is "largest" rather than llama.cpp's
///      "first listed" (alphabetical) which would systematically pick BF16.
///
/// The selection lives in the Rust core (`net.resolveFromSiblings`); this type
/// does the transport (fetch the sibling list from the HF API) and maps the
/// core's structured result/error back onto the Swift surface.
enum HFRepoResolver {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "HFRepoResolver")

  /// Everything needed to start a download for a resolved deeplink.
  struct Resolved {
    /// Stable id from `Model.makeId` — the verbatim `"{org}/{repo}:{TAG}"`.
    /// Matches the id shape `HFCache.buildSideloadedEntry` produces, so
    /// post-install the row keeps the same identity without any handoff.
    let modelId: String
    /// Main GGUF URL (goes into `Model.downloadUrl`).
    let mainUrl: URL
    /// Sharded parts, if any (`-00002-of-NNNNN.gguf`, ...).
    let additionalParts: [URL]
    /// Optional mmproj sidecar.
    let mmprojUrl: URL?
    /// Optional draft-head sidecar (`dflash-….gguf`, else `mtp-….gguf`),
    /// quant-matched to the main.
    let draftUrl: URL?
    /// Size in bytes if the HF API disclosed it (aggregated across main+shards+mmproj).
    /// 0 when the API didn't return per-file sizes. We start the download either way;
    /// `fetchHFDownloadPlan` does HEAD requests that produce real byte counts.
    let approximateBytes: Int64
  }

  /// User-facing errors. All map to `NSAlert` surface text in `DeeplinkHandler`.
  enum ResolveError: LocalizedError {
    case invalidQuant(String)
    case repoNotFound(String)
    case gated(String)
    case networkUnavailable(String)
    case noGgufFiles(String)
    case quantNotFound(repo: String, quant: String)
    case shardMissing(repo: String, shard: String)
    case noCompatibleFile(repo: String)

    var errorDescription: String? {
      switch self {
      case .invalidQuant(let q):
        return "“\(q)” isn’t a known GGUF quantization label."
      case .repoNotFound(let repo):
        return "Hugging Face returned 404 for \(repo)."
      case .gated(let repo):
        return "\(repo) is gated or private."
      case .networkUnavailable:
        return "Couldn’t reach Hugging Face."
      case .noGgufFiles(let repo):
        return "\(repo) doesn’t contain any GGUF files."
      case .quantNotFound(let repo, let quant):
        return "No \(quant) file found in \(repo)."
      case .shardMissing(let repo, let shard):
        return "Shard \(shard) is missing from \(repo)."
      case .noCompatibleFile(let repo):
        return "No quantization in \(repo) fits this Mac’s memory budget."
      }
    }

    var recoverySuggestion: String? {
      switch self {
      case .gated:
        return "Open Settings > Advanced and paste a Hugging Face access token that has access to this repo."
      case .networkUnavailable:
        return "Check your internet connection and try again."
      case .noCompatibleFile:
        return "Open the repo page on Hugging Face and pick a smaller quant manually."
      default:
        return nil
      }
    }
  }

  /// Resolves against the live HF API. Authenticates with the configured HF
  /// token so gated repos with a valid token succeed.
  static func resolve(
    repo: String,
    quant: String?,
    token: String?
  ) async throws -> Resolved {
    // Up-front validation: a malformed but present quant is a hard reject.
    // Otherwise a tampered link could silently install a different file by
    // falling through to default-quant resolution. (The core re-checks; we do
    // it here too so a bad quant never costs a network round-trip.)
    if let quant, GGUFQuant.parseLabel(quant) == nil {
      throw ResolveError.invalidQuant(quant)
    }

    let siblings = try await fetchSiblings(repo: repo, token: token)

    var args: [String: Any] = [
      "repo": repo,
      "budgetMb": Model.deviceMemoryBudgetMb,
      "siblings": siblings.map { sib -> [String: Any] in
        var d: [String: Any] = ["rfilename": sib.rfilename]
        if let size = sib.size { d["size"] = size }
        return d
      },
    ]
    if let quant { args["requestedQuant"] = quant }

    guard let out = LlamaCore.callJSON("net.resolveFromSiblings", args) else {
      throw ResolveError.networkUnavailable(repo)
    }
    if out["error"] != nil {
      throw resolveError(from: out)
    }

    guard let modelId = out["modelId"] as? String,
      let mainUrl = (out["mainUrl"] as? String).flatMap(URL.init(string:))
    else {
      throw ResolveError.networkUnavailable(repo)
    }

    return Resolved(
      modelId: modelId,
      mainUrl: mainUrl,
      additionalParts: (out["additionalParts"] as? [String] ?? []).compactMap(URL.init(string:)),
      mmprojUrl: (out["mmprojUrl"] as? String).flatMap(URL.init(string:)),
      draftUrl: (out["draftUrl"] as? String).flatMap(URL.init(string:)),
      approximateBytes: (out["approximateBytes"] as? NSNumber)?.int64Value ?? 0
    )
  }

  /// Rebuilds the typed `ResolveError` from the core's structured error object
  /// (`resolve_error_to_value`), so the UI keeps its original title/recovery.
  private static func resolveError(from out: [String: Any]) -> ResolveError {
    let kind = out["error"] as? String ?? ""
    let repo = out["repo"] as? String ?? ""
    switch kind {
    case "invalidQuant":
      return .invalidQuant(out["quant"] as? String ?? "")
    case "repoNotFound":
      return .repoNotFound(repo)
    case "gated":
      return .gated(repo)
    case "noGgufFiles":
      return .noGgufFiles(repo)
    case "quantNotFound":
      return .quantNotFound(repo: repo, quant: out["quant"] as? String ?? "")
    case "shardMissing":
      return .shardMissing(repo: repo, shard: out["shard"] as? String ?? "")
    case "noCompatibleFile":
      return .noCompatibleFile(repo: repo)
    default:
      return .networkUnavailable(repo)
    }
  }

  // MARK: - HF API

  /// Minimal sibling representation — `rfilename` is repo-relative path
  /// (e.g. `Q4_K_M/model.gguf`), `size` is bytes when the API supplies it.
  struct Sibling: Decodable {
    let rfilename: String
    let size: Int64?
    private enum CodingKeys: String, CodingKey {
      case rfilename, size
    }
  }

  private struct ModelInfoResponse: Decodable {
    let siblings: [Sibling]?
  }

  private static func fetchSiblings(repo: String, token: String?) async throws -> [Sibling] {
    // `?blobs=true` asks HF to populate `siblings[].size` alongside filenames.
    // Without it, many responses omit sizes and we'd have to HEAD every
    // candidate just to rank them.
    guard let url = URL(string: "https://huggingface.co/api/models/\(repo)?blobs=true") else {
      throw ResolveError.repoNotFound(repo)
    }
    var req = HFRequest.make(url, token: token)
    req.setValue("application/json", forHTTPHeaderField: "Accept")

    let data: Data
    let response: URLResponse
    do {
      (data, response) = try await URLSession.shared.data(for: req)
    } catch {
      logger.error("HF API fetch failed for \(repo): \(error.localizedDescription)")
      throw ResolveError.networkUnavailable(repo)
    }

    guard let http = response as? HTTPURLResponse else {
      throw ResolveError.networkUnavailable(repo)
    }
    switch http.statusCode {
    case 200...299: break
    case 401, 403: throw ResolveError.gated(repo)
    case 404: throw ResolveError.repoNotFound(repo)
    default:
      logger.error("HF API \(http.statusCode) for \(repo)")
      throw ResolveError.networkUnavailable(repo)
    }

    guard let decoded = try? JSONDecoder().decode(ModelInfoResponse.self, from: data),
      let siblings = decoded.siblings
    else {
      return []
    }
    return siblings
  }
}
