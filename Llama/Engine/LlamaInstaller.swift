import CryptoKit
import Darwin
import Foundation
import os.log

/// Installs and updates the app-managed `llama` CLI from the official GitHub
/// release archive. Each build lives in its own directory under `~/.llama-app`;
/// `~/.llama-app/llama` is an atomically switched symlink to that build, and
/// `~/.local/bin/llama` points at the stable symlink for terminal use.
///
/// This only ever writes the managed path. A Homebrew (or other unmanaged)
/// install is never touched -- see `LlamaBinaries`.
///
/// Artifact URLs and managed-binary promotion live in the Rust core
/// (`engine.artifactUrl`, `engine.promoteStaged`). This type handles the async
/// download and extraction because the core has no async transport.
enum LlamaInstaller {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "LlamaInstaller")

  /// The real binary path (also what `LlamaBinaries` resolves as `.managed`).
  private static var llamaPath: String { LlamaBinaries.managedPath }

  /// Install dir the app manages. Release archives are unpacked under `releases/`.
  private static var installDir: String {
    (LlamaBinaries.managedPath as NSString).deletingLastPathComponent
  }
  private static func releaseDir(version: String) -> String {
    installDir + "/releases/llama-\(version)"
  }

  /// Where a background update is staged. The live path never changes under a
  /// running server; a staged binary is promoted (renamed over `llamaPath`) at
  /// the next app launch, before the server starts.
  static var stagedPath: String { LlamaBinaries.managedPath + ".staged" }

  enum InstallError: Error, LocalizedError {
    case downloadFailed(String)
    case installFailed(String)

    var errorDescription: String? {
      switch self {
      case .downloadFailed(let detail):
        return "Download failed: \(detail)"
      case .installFailed(let detail):
        return "Couldn't install the llama binary: \(detail)"
      }
    }
  }

  // MARK: - Public API

  /// Installs (or replaces) the app-managed `llama` binary at `version`.
  /// With `staged: true` the binary lands at `stagedPath` instead of the live
  /// path, to be promoted at the next launch -- used for background updates so
  /// the binary a running server spawns model instances from never changes
  /// underneath it.
  static func install(version: String, staged: Bool = false) async throws {
    logger.info("Installing llama \(version, privacy: .public)\(staged ? " (staged)" : "")")

    let artifactUrl: String =
      LlamaCore.callValue("engine.artifactUrl", "url", ["version": version]) ?? ""
    let archive = try await download(
      artifactUrl,
      expectedSHA256: version == LlamaBinaries.targetVersion.tag
        ? LlamaBinaries.targetArtifactSHA256 : nil)
    defer { try? FileManager.default.removeItem(at: archive) }

    let bundle = try extractReleaseArchive(archive, version: version)
    let binary = bundle + "/llama"
    let link = staged ? stagedPath : llamaPath
    try atomicallyPoint(link, to: binary)
    if !staged { linkOnPath() }

    logger.info("Installed llama \(version, privacy: .public) at \(binary, privacy: .public)")
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

  /// Unpacks the versioned upstream archive into a temporary directory, verifies
  /// that its executable reports the requested build, then moves the complete
  /// bundle into its immutable release directory. Keeping dylibs beside the
  /// binary is required by the official macOS build and makes staged updates safe.
  private static func extractReleaseArchive(_ archive: URL, version: String) throws -> String {
    let fileManager = FileManager.default
    let releases = installDir + "/releases"
    let staging = installDir + "/.extract-\(UUID().uuidString)"
    let expectedBundleName = "llama-\(version)"
    let extractedBundle = staging + "/" + expectedBundleName
    let destination = releaseDir(version: version)
    defer { try? fileManager.removeItem(atPath: staging) }

    do {
      try fileManager.createDirectory(atPath: staging, withIntermediateDirectories: true)
      try fileManager.createDirectory(atPath: releases, withIntermediateDirectories: true)
      let process = Process()
      process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
      process.arguments = ["-xzf", archive.path, "-C", staging]
      try process.run()
      process.waitUntilExit()
      guard process.terminationStatus == 0,
        fileManager.isExecutableFile(atPath: extractedBundle + "/llama")
      else {
        throw InstallError.installFailed("The llama.cpp release archive is incomplete.")
      }

      guard LlamaBinaries.readVersion(at: extractedBundle + "/llama")?.tag == version else {
        throw InstallError.installFailed("The release archive did not contain llama \(version).")
      }

      if fileManager.fileExists(atPath: destination) {
        try fileManager.removeItem(atPath: destination)
      }
      try fileManager.moveItem(atPath: extractedBundle, toPath: destination)
      return destination
    } catch let error as InstallError {
      throw error
    } catch {
      throw InstallError.installFailed(error.localizedDescription)
    }
  }

  /// Atomically switches a stable managed path (or staged path) to a release
  /// executable. The binary itself and all dylibs remain in the versioned dir.
  private static func atomicallyPoint(_ link: String, to binary: String) throws {
    let fileManager = FileManager.default
    let directory = (link as NSString).deletingLastPathComponent
    try fileManager.createDirectory(atPath: directory, withIntermediateDirectories: true)
    let temporaryLink = link + ".tmp-\(UUID().uuidString)"
    do {
      try fileManager.createSymbolicLink(atPath: temporaryLink, withDestinationPath: binary)
      let result = temporaryLink.withCString { source in
        link.withCString { destination in rename(source, destination) }
      }
      guard result == 0 else {
        let code = errno
        throw InstallError.installFailed(String(cString: strerror(code)))
      }
    } catch {
      try? fileManager.removeItem(atPath: temporaryLink)
      throw error
    }
  }

  /// Points `~/.local/bin/llama` at the stable managed symlink so `llama` is
  /// available in a terminal. Best-effort; the core never clobbers a real file.
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

  /// Downloads an artifact to a temp file and verifies it against an explicit
  /// release digest or Hugging Face's advertised blob SHA256. A mismatch aborts
  /// the install before any downloaded executable is unpacked or promoted.
  ///
  /// Kept in Swift for `URLSession`'s async download (the core has no async
  /// transport).
  private static func download(
    _ urlString: String, expectedSHA256: String? = nil
  ) async throws -> URL {
    guard let url = URL(string: urlString) else {
      throw InstallError.downloadFailed("bad URL: \(urlString)")
    }
    let token = UserSettings.hfToken
    let expected: String?
    if let expectedSHA256 {
      expected = expectedSHA256
    } else if url.host?.hasSuffix("huggingface.co") == true {
      expected = await HFCache.fetchFileMetadata(for: [url], token: token)[url]?.blobHash
    } else {
      expected = nil
    }

    let request = HFRequest.make(url, token: token)
    let (tempURL, response) = try await URLSession.shared.download(for: request)
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
      let code = (response as? HTTPURLResponse)?.statusCode ?? -1
      throw InstallError.downloadFailed("\(urlString) (HTTP \(code))")
    }

    if let expected {
      let computed = try sha256(of: tempURL)
      guard digestMatches(computed: computed, expected: expected) else {
        try? FileManager.default.removeItem(at: tempURL)
        throw InstallError.downloadFailed(
          "\(url.lastPathComponent) failed its integrity check (SHA256 mismatch)")
      }
    }
    return tempURL
  }

  /// Whether a downloaded file's digest matches the advertised or pinned hash.
  /// A nil `expected` means the source advertised no hash, so nothing to check.
  static func digestMatches(computed: String, expected: String?) -> Bool {
    guard let expected else { return true }
    return computed == expected.lowercased()
  }

  /// Streams `fileURL` through SHA256 and returns lowercase hex. Runs off the
  /// main actor via the caller.
  private static func sha256(of fileURL: URL) throws -> String {
    var hasher = SHA256()
    try HFCache.feedHasher(&hasher, from: fileURL)
    return HFCache.hex(hasher.finalize())
  }
}
