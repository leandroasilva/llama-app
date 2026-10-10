import CryptoKit
import Foundation
import os.log

/// Utility for working with the standard HuggingFace Hub cache layout.
///
/// Layout:
/// ```
/// ~/.cache/huggingface/hub/
/// └── models--{org}--{repo}/
///     ├── blobs/
///     │   └── {sha256}              # actual file, named by content hash
///     ├── refs/
///     │   └── main                  # text file containing commit hash
///     └── snapshots/
///         └── {commit}/
///             └── filename.gguf -> ../../blobs/{sha256}
/// ```
///
/// Every path/scan/delete operation delegates to the Rust core (`hfCache.*`),
/// which owns the layout rules so the app and the core can't diverge. What
/// stays here is the platform-bound work the core can't do: URLSession HEAD
/// requests and CryptoKit hashing.
enum HFCache {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "HFCache")

  /// Name of the hidden dir under the HF cache that holds in-progress `.partial`
  /// files. Tied to the app's "llama" identity; `RenameMigration` moves the
  /// pre-rename `.llamabarn-partial` dir here on first launch.
  static let partialRootDirName = ".llama-partial"

  // MARK: - Path Helpers (pure, no I/O)

  /// Parses a HF download URL and returns the repo directory name.
  /// e.g. `https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/file.gguf`
  /// → `"models--unsloth--Qwen3.5-2B-GGUF"`
  static func repoDirName(from url: URL) -> String? {
    LlamaCore.callValue("hfCache.repoDirName", "name", ["url": url.absoluteString])
  }

  /// The repo-relative path a HF download URL points at (everything past
  /// `resolve/{branch}/`), preserving subdir structure. e.g.
  /// `https://huggingface.co/org/repo/resolve/main/Q4_K_M/model.gguf`
  /// → `"Q4_K_M/model.gguf"`.
  /// Returns nil for URLs that don't match the HF resolve shape.
  static func repoRelativePath(from url: URL) -> String? {
    LlamaCore.callValue("hfCache.repoRelativePath", "path", ["url": url.absoluteString])
  }

  /// Directory that holds in-progress `.partial` files for a model.
  /// Lives under the HF cache so promoting `.partial` → `blobs/<hash>` stays on the
  /// same filesystem (moveItem is atomic).
  static func partialDir(cacheDir: URL, modelId: String) -> URL {
    pathURL("hfCache.partialDir", ["cacheDir": cacheDir.path, "modelId": modelId])
  }

  /// Path to an in-progress `.partial` file for a single remote file.
  static func partialPath(cacheDir: URL, modelId: String, filename: String) -> URL {
    pathURL(
      "hfCache.partialPath",
      ["cacheDir": cacheDir.path, "modelId": modelId, "filename": filename])
  }

  /// Removes the model's partial directory (and any files under it).
  /// Silently ignores missing dirs — used by cancel/delete/cleanup paths.
  static func removePartials(cacheDir: URL, modelId: String) {
    _ = LlamaCore.call("hfCache.removePartials", LlamaCore.encode([
      "cacheDir": cacheDir.path, "modelId": modelId,
    ]))
  }

  /// Sum of `.partial` file sizes in a single model's staging dir.
  /// Returns 0 if the dir doesn't exist or contains no `.partial` files.
  ///
  /// Walks recursively — some HF repos nest GGUFs in per-quant subdirs
  /// (e.g. `Q4_K_M/model.gguf`), and the partial layout mirrors the repo
  /// layout, so partials can live one level deep.
  static func partialBytes(cacheDir: URL, modelId: String) -> Int64 {
    let out = LlamaCore.callJSON("hfCache.partialBytes", [
      "cacheDir": cacheDir.path, "modelId": modelId,
    ])
    return (out?["bytes"] as? NSNumber)?.int64Value ?? 0
  }

  /// File name of the per-download placeholder dropped next to a model's
  /// `.partial` files. Hidden (leading dot) and not a `.partial`, so
  /// `partialBytes` ignores it; it rides inside the partial dir so the existing
  /// `removePartials` cleanup disposes of it on completion/cancel.
  static let placeholderFileName = ".model.json"

  /// Path to a download's placeholder file (the serialized `Model`). Written
  /// when a download starts so the app can rebuild the paused row at launch
  /// without a network resolve; see `scanPlaceholders`.
  static func placeholderURL(cacheDir: URL, modelId: String) -> URL {
    pathURL("hfCache.placeholderURL", ["cacheDir": cacheDir.path, "modelId": modelId])
  }

  /// Finds every download placeholder under the partial root — one per
  /// interrupted/paused download from a previous session. Used at launch to
  /// rehydrate paused rows. Returns the placeholder file URLs; the caller
  /// decodes the `Model` (HFCache stays model-agnostic).
  static func scanPlaceholders(cacheDir: URL) -> [URL] {
    let out = LlamaCore.callJSON("hfCache.scanPlaceholders", ["cacheDir": cacheDir.path])
    let paths = out?["paths"] as? [String] ?? []
    return paths.map { URL(fileURLWithPath: $0) }
  }

  /// Cleans up `<partialRootDirName>/<id>` subdirs for ids that are already
  /// installed (e.g. a deeplink download that landed but left its partial dir
  /// behind). Interrupted (not-yet-installed) downloads are kept and surfaced
  /// as paused rows at launch via `scanPlaceholders`.
  static func cleanInstalledPartials(cacheDir: URL, installedIds: Set<String>) {
    _ = LlamaCore.call("hfCache.cleanInstalledPartials", LlamaCore.encode([
      "cacheDir": cacheDir.path, "installedIds": Array(installedIds),
    ]))
  }

  /// Returns true if `filename` (repo-relative) exists in any snapshot commit
  /// of the repo — i.e. the file is already installed in the cache.
  static func snapshotFileExists(cacheDir: URL, repoDir: String, filename: String) -> Bool {
    let out = LlamaCore.callJSON("hfCache.snapshotFileExists", [
      "cacheDir": cacheDir.path, "repoDir": repoDir, "filename": filename,
    ])
    return out?["value"] as? Bool ?? false
  }

  // MARK: - API Calls

  /// Metadata returned by a HEAD request to a HF file URL.
  struct FileMetadata {
    /// Content hash (SHA256) from X-Linked-Etag or ETag header.
    /// Nil if neither header contains a valid SHA256.
    let blobHash: String?
    /// Repo commit hash from the X-Repo-Commit header.
    /// Nil if the header is missing.
    let commitHash: String?
  }

  /// Fetches blob hash and commit hash for multiple files via HEAD requests.
  ///
  /// HF serves `X-Linked-Etag` (blob SHA256) and `X-Repo-Commit` (commit hash) in
  /// the response to the resolve URL. We use a same-host redirect delegate to prevent
  /// following redirects to the CDN, which would lose these headers.
  ///
  /// Uses a single URLSession for all requests (sessions are heavyweight).
  static func fetchFileMetadata(
    for urls: [URL], token: String?
  ) async -> [URL: FileMetadata] {
    let delegate = SameHostRedirectDelegate()
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
    defer { session.finishTasksAndInvalidate() }

    var results: [URL: FileMetadata] = [:]

    for url in urls {
      var request = HFRequest.make(url, token: token)
      request.httpMethod = "HEAD"

      guard let (_, response) = try? await session.data(for: request),
        let httpResponse = response as? HTTPURLResponse,
        (200...399).contains(httpResponse.statusCode)
      else { continue }

      // Extract blob hash from X-Linked-Etag (preferred) or ETag (fallback)
      let blobHash: String? = {
        if let etag = httpResponse.value(forHTTPHeaderField: "X-Linked-Etag") {
          return etag.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        if let etag = httpResponse.value(forHTTPHeaderField: "ETag") {
          let cleaned =
            etag
            .replacingOccurrences(of: "W/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
          // Only use if it looks like a SHA256 (64 hex chars)
          if cleaned.count == 64, cleaned.allSatisfy({ $0.isHexDigit }) {
            return cleaned
          }
        }
        return nil
      }()

      let commitHash = httpResponse.value(forHTTPHeaderField: "X-Repo-Commit")
      results[url] = FileMetadata(blobHash: blobHash, commitHash: commitHash)
    }

    return results
  }

  // MARK: - File Operations

  /// Writes a downloaded file into the HF cache layout.
  ///
  /// 1. Creates directory structure (blobs/, snapshots/{commit}/, refs/)
  /// 2. Moves temp file → blobs/{sha256}
  /// 3. Creates symlink snapshots/{commit}/{filename} → ../../blobs/{sha256}
  /// 4. Writes refs/main with the commit hash
  static func writeBlobAndLink(
    cacheDir: URL,
    repoDir: String,
    commit: String,
    blobHash: String,
    filename: String,
    from tempFile: URL
  ) throws {
    let out = LlamaCore.callJSON("hfCache.writeBlobAndLink", [
      "cacheDir": cacheDir.path,
      "repoDir": repoDir,
      "commit": commit,
      "blobHash": blobHash,
      "filename": filename,
      "tempFile": tempFile.path,
    ])
    guard out?["ok"] as? Bool == true else {
      throw HFCacheError(
        message: (out?["error"] as? String) ?? "hfCache.writeBlobAndLink failed")
    }
    logger.info("Wrote HF cache: \(repoDir)/blobs/\(blobHash) + snapshot symlink for \(filename)")
  }

  /// Lowercase hex of a SHA256 digest -- the form HF names blobs by.
  ///
  /// The resumable download path hashes incrementally with CryptoKit's
  /// `SHA256`: bytes are streamed into the hasher as they're written to the
  /// `.partial` file (and any existing prefix is re-hashed once at resume time,
  /// see `feedHasher`), so the digest is ready at completion without a second
  /// full-file pass.
  static func hex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
  }

  /// Feeds the entire contents of `fileURL` into `hasher` in 1 MB chunks.
  /// Used on resume to reconstruct the running hash over the existing `.partial` prefix.
  static func feedHasher(_ hasher: inout SHA256, from fileURL: URL) throws {
    let handle = try FileHandle(forReadingFrom: fileURL)
    defer { try? handle.close() }
    let chunkSize = 1_048_576  // 1 MB
    while autoreleasepool(invoking: {
      let data = handle.readData(ofLength: chunkSize)
      guard !data.isEmpty else { return false }
      hasher.update(data: data)
      return true
    }) {}
  }

  // MARK: - Deletion

  /// Deletes a model's files from the HF cache.
  /// Removes blobs (resolved from symlinks), snapshot symlinks, and cleans up empty dirs.
  static func deleteModelFiles(cacheDir: URL, repoDir: String, paths: ResolvedPaths) throws {
    var pathArgs: [String: Any] = [
      "modelFile": paths.modelFile,
      "additionalParts": paths.additionalParts,
      "usesMTP": paths.usesMTP,
      "hfRepoDirName": paths.hfRepoDirName,
    ]
    if let mmprojFile = paths.mmprojFile { pathArgs["mmprojFile"] = mmprojFile }
    if let sidecar = paths.draftSidecarFile { pathArgs["draftSidecarFile"] = sidecar }

    let out = LlamaCore.callJSON("hfCache.deleteModelFiles", [
      "cacheDir": cacheDir.path,
      "repoDir": repoDir,
      "paths": pathArgs,
    ])
    guard out?["ok"] as? Bool == true else {
      throw HFCacheError(
        message: (out?["error"] as? String) ?? "hfCache.deleteModelFiles failed")
    }
  }

  // MARK: - Discovery

  /// Scans the HF cache for GGUF files and builds a `Model` + `ResolvedPaths`
  /// per discovered file (or shard group).
  ///
  /// For each GGUF, metadata is parsed from the repo directory name and
  /// filename using the same approach as llama.cpp's WebUI model selector.
  /// Split GGUFs (e.g. -00001-of-00003.gguf) are grouped as single entries.
  static func scanForSideloaded(
    cacheDir: URL
  ) -> [(entry: Model, paths: ResolvedPaths)] {
    let out = LlamaCore.callJSON("hfCache.scanForSideloaded", ["cacheDir": cacheDir.path])
    guard let entries = out?["entries"] as? [[String: Any]] else { return [] }

    return entries.compactMap { e in
      guard let id = e["id"] as? String,
        let ctxWindow = (e["ctxWindow"] as? NSNumber)?.intValue,
        let pathsDict = e["paths"] as? [String: Any],
        let paths = resolvedPaths(from: pathsDict)
      else { return nil }

      let entry = Model(
        id: id,
        ctxWindow: ctxWindow,
        fileSize: (e["fileSize"] as? NSNumber)?.int64Value ?? 0,
        ctxBytesPer1kTokens: 0,
        downloadUrl: URL(string: "file:///")!,
        scannedDraftHead: (e["draftHead"] as? String).flatMap(DraftHead.init(rawValue:)),
        isDecisionModel: e["isDecisionModel"] as? Bool ?? false
      )
      return (entry: entry, paths: paths)
    }
  }

  // MARK: - FFI Decoding Helpers

  /// Decodes an `hfCache.*` response carrying a single `path` field into a URL.
  private static func pathURL(_ method: String, _ args: [String: Any]) -> URL {
    guard let path = LlamaCore.callJSON(method, args)?["path"] as? String else {
      return URL(fileURLWithPath: "")
    }
    return URL(fileURLWithPath: path)
  }

  /// Rebuilds a `ResolvedPaths` from a `resolvedPathsToValue` object.
  private static func resolvedPaths(from d: [String: Any]) -> ResolvedPaths? {
    ResolvedPaths(ffi: d)
  }
}

/// Error surfaced when a Rust `hfCache.*` filesystem operation fails.
struct HFCacheError: Error, CustomStringConvertible {
  let message: String
  var description: String { message }
}

// MARK: - Same-Host Redirect Delegate

/// URLSession delegate that blocks cross-host redirects.
/// HF redirects file URLs to a CDN for the actual download. The CDN response
/// won't have HF-specific headers (X-Linked-Etag, X-Repo-Commit). By blocking
/// the redirect, the HEAD request returns HF's 302 response with those headers intact.
private class SameHostRedirectDelegate: NSObject, URLSessionTaskDelegate {
  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    // Allow same-host redirects (e.g., HTTP → HTTPS), block cross-host (HF → CDN)
    if task.originalRequest?.url?.host == request.url?.host {
      completionHandler(request)
    } else {
      completionHandler(nil)
    }
  }
}
