import Darwin
import Foundation
import os.log

/// Installs and updates the app-managed `llama` CLI by fetching prebuilt binaries
/// from the same Hugging Face bucket the official `install.sh` uses, laid out the same way:
/// the real binary at `~/.llama-app/llama` plus a `~/.local/bin/llama` symlink
/// so `llama` works from a terminal too.
///
/// macOS ships no zstd and the bucket serves the binary zstd-compressed, so we
/// fetch the bucket's small standalone `unzstd` helper (cached alongside the
/// binary) and pipe the download through it.
///
/// This only ever writes the managed path. A Homebrew (or other unmanaged)
/// install is never touched -- see `LlamaBinaries`.
///
/// The Metal-config mapping, artifact URLs, decompression, atomic install and
/// PATH symlink all live in the Rust core (`engine.metalConfig`,
/// `engine.artifactUrl`, `engine.unzstdUrl`, `engine.decompress`,
/// `engine.atomicInstall`, `engine.ensurePathSymlink`, `engine.promoteStaged`).
/// This type keeps the async-download orchestration (`install`) and the
/// `URLSession` download, which the core has no async transport for.
enum LlamaInstaller {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "LlamaInstaller")

  /// The real binary path (also what `LlamaBinaries` resolves as `.managed`).
  private static var llamaPath: String { LlamaBinaries.managedPath }

  /// Install dir the app manages. Holds the binary and the cached `unzstd` helper.
  private static var installDir: String {
    (LlamaBinaries.managedPath as NSString).deletingLastPathComponent
  }
  private static var unzstdPath: String { installDir + "/unzstd" }

  /// Where a background update is staged. The live path never changes under a
  /// running server; a staged binary is promoted (renamed over `llamaPath`) at
  /// the next app launch, before the server starts.
  static var stagedPath: String { LlamaBinaries.managedPath + ".staged" }

  enum InstallError: Error, LocalizedError {
    case unsupportedHardware(String)
    case downloadFailed(String)
    case installFailed(String)

    var errorDescription: String? {
      switch self {
      case .unsupportedHardware(let detail):
        return "No prebuilt llama binary for this Mac (\(detail))."
      case .downloadFailed(let detail):
        return "Download failed: \(detail)"
      case .installFailed(let detail):
        return "Couldn't install the llama binary: \(detail)"
      }
    }
  }

  /// A failure the core already formatted: its `message()` text matches the
  /// previous Swift `InstallError.errorDescription`, so it's surfaced verbatim
  /// rather than re-wrapped.
  struct InstallFailure: Error, LocalizedError {
    let kind: String
    let message: String
    var errorDescription: String? { message }
  }

  // MARK: - Public API

  /// Installs (or replaces) the app-managed `llama` binary at `version`.
  /// With `staged: true` the binary lands at `stagedPath` instead of the live
  /// path, to be promoted at the next launch -- used for background updates so
  /// the binary a running server spawns model instances from never changes
  /// underneath it.
  static func install(version: String, staged: Bool = false) async throws {
    logger.info("Installing llama \(version, privacy: .public)\(staged ? " (staged)" : "")")

    let config = try metalConfig()
    try await ensureUnzstd(version: version)

    let artifactUrl: String =
      LlamaCore.callValue("engine.artifactUrl", "url", ["version": version, "config": config]) ?? ""
    let zstURL = try await download(artifactUrl)
    defer { try? FileManager.default.removeItem(at: zstURL) }

    let dest = staged ? stagedPath : llamaPath
    try decompress(zst: zstURL, to: dest)
    if !staged { linkOnPath() }

    logger.info("Installed llama \(version, privacy: .public) at \(dest, privacy: .public)")
  }

  /// Promotes a staged binary into the live path if it matches `target`, else
  /// discards it (the pin moved on since it was staged). Call at launch, before
  /// the server starts. Runs a `version` subprocess, so call off the main
  /// thread.
  static func promoteStaged(target: LlamaVersion) {
    let out = LlamaCore.callJSON("engine.promoteStaged", [
      "staged": stagedPath, "live": llamaPath, "targetBuild": target.build,
      "home": NSHomeDirectory(),
    ])
    switch out?["outcome"] as? String {
    case "discardedStale":
      logger.info("Discarding stale staged llama binary")
    case "failed":
      logger.error(
        "Couldn't promote staged llama binary: \(out?["message"] as? String ?? "", privacy: .public)"
      )
    case "promoted":
      logger.info("Promoted staged llama binary \(target.tag, privacy: .public)")
    default:
      break  // nothingStaged
    }
  }

  // MARK: - Steps

  /// Maps this Mac's CPU to a Metal bucket config (`m1`…`m5`, `a18`), mirroring
  /// `install.sh`, which buckets every M-series variant (Pro/Max/Ultra) by
  /// generation number and also accepts the A18 (non-M MacBook). The sysctl read
  /// stays here; the brand→config mapping lives in the core.
  private static func metalConfig() throws -> String {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    guard size > 0 else { throw InstallError.unsupportedHardware("unknown CPU") }
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
    let brand = String(cString: buf)

    guard let config: String = LlamaCore.callValue("engine.metalConfig", "config", ["brand": brand])
    else {
      throw InstallError.unsupportedHardware(brand)
    }
    return config
  }

  /// Ensures the `unzstd` helper is present, fetching it once and caching it
  /// next to the binary. Subsequent installs/updates reuse it.
  private static func ensureUnzstd(version: String) async throws {
    if FileManager.default.isExecutableFile(atPath: unzstdPath) { return }
    let url: String = LlamaCore.callValue("engine.unzstdUrl", "url", ["version": version]) ?? ""
    let tmp = try await download(url)
    defer { try? FileManager.default.removeItem(at: tmp) }
    try atomicInstall(from: tmp, to: unzstdPath)
  }

  /// Decompresses `zst` through the `unzstd` helper and atomically swaps the
  /// result into `dest`. The temp output lives in the install dir so the final
  /// `rename(2)` is a same-volume atomic replace -- a crash never leaves a
  /// half-written binary in place. Runs in the core (`engine.decompress`).
  private static func decompress(zst: URL, to dest: String) throws {
    let out = LlamaCore.callJSON("engine.decompress", [
      "unzstd": unzstdPath, "zst": zst.path, "dest": dest,
    ])
    try checkInstallResult(out)
  }

  /// Mirrors `install.sh`'s `ln -sf`: point `~/.local/bin/llama` at our
  /// binary so it's available in a terminal. Best-effort -- the app resolves the
  /// real path directly, so a failure here doesn't break the app. The core
  /// (`engine.ensurePathSymlink`) won't clobber a real file a user may have
  /// placed there; only absent or symlink entries are (re)written.
  private static func linkOnPath() {
    let out = LlamaCore.callJSON("engine.ensurePathSymlink", [
      "home": NSHomeDirectory(), "llamaPath": llamaPath,
    ])
    switch out?["outcome"] as? String {
    case "leftExisting":
      logger.info("Leaving existing non-symlink at ~/.local/bin/llama")
    case "failed":
      logger.error(
        "Couldn't create PATH symlink: \(out?["message"] as? String ?? "", privacy: .public)")
    default:
      break
    }
  }

  // MARK: - Helpers

  /// Downloads a URL to a temp file, following HF's redirect to the CDN. Kept in
  /// Swift for `URLSession`'s async download (the core has no async transport).
  private static func download(_ urlString: String) async throws -> URL {
    guard let url = URL(string: urlString) else {
      throw InstallError.downloadFailed("bad URL: \(urlString)")
    }
    let request = HFRequest.make(url, token: UserSettings.hfToken)
    let (tempURL, response) = try await URLSession.shared.download(for: request)
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
      let code = (response as? HTTPURLResponse)?.statusCode ?? -1
      throw InstallError.downloadFailed("\(urlString) (HTTP \(code))")
    }
    return tempURL
  }

  /// Copies `src` (possibly on another volume) onto the destination volume, then
  /// atomically renames it into place as an executable. Runs in the core
  /// (`engine.atomicInstall`).
  private static func atomicInstall(from src: URL, to dest: String) throws {
    let out = LlamaCore.callJSON("engine.atomicInstall", ["src": src.path, "dest": dest])
    try checkInstallResult(out)
  }

  /// Throws an `InstallFailure` if a core install-result object carries an
  /// `error`; no-op on `{"ok":true}` or a missing response.
  private static func checkInstallResult(_ out: [String: Any]?) throws {
    guard let error = out?["error"] as? String else { return }
    throw InstallFailure(kind: error, message: out?["message"] as? String ?? error)
  }
}
