import Foundation
import XCTest

@testable import Llama

/// Fase 4 equivalence: Engine (LlamaBinaries / LlamaInstaller leaves /
/// MemProfile) against the shared `Fixtures/engine.json` corpus and the
/// `Fixtures/engine/*` fake subprocesses.
///
/// Parity note: `LlamaBinaries.resolve` reads `NSHomeDirectory()`,
/// `LlamaInstaller.metalConfig` reads the sysctl brand, and
/// `MemProfileRunner.probeTotalMib/run` spawn the real `llama` — none of those
/// are injectable in the Swift originals. So the ambiguity-free logic is pinned
/// here through the FFI (the Rust `phase4.rs` asserts the same numbers with the
/// same fixture), and the cache JSON bytes are checked both directions against
/// the Swift `Codable` original so old caches stay readable.
final class Phase4FixtureTests: XCTestCase {

  // MARK: - Fixture loading + FFI helpers

  private static let fixturesDir = Fixtures.directory

  private func loadJSON(_ name: String) throws -> [String: Any] {
    let url = Self.fixturesDir.appendingPathComponent(name)
    let data = try Data(contentsOf: url)
    return try XCTUnwrap(
      JSONSerialization.jsonObject(with: data) as? [String: Any], "top-level dict")
  }

  private func jsonString(_ object: Any) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: object)
    return String(decoding: data, as: UTF8.self)
  }

  private func decode(_ raw: String?) throws -> [String: Any] {
    let raw = try XCTUnwrap(raw, "ffi returned nil")
    return try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any], "ffi dict")
  }

  /// Like `decode` but tolerates a top-level null / array (used by
  /// `engine.binariesResolve`, which can resolve to nothing, and `engine.fit`,
  /// which returns null for a degenerate fit).
  private func decodeAny(_ raw: String?) throws -> Any {
    let raw = try XCTUnwrap(raw, "ffi returned nil")
    if raw == "null" { return NSNull() }
    if let value = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) {
      return value
    }
    return NSNull()
  }

  private func call(_ method: String, _ args: [String: Any]) throws -> [String: Any] {
    try decode(LlamaCore.call(method, try jsonString(args)))
  }

  private func callAny(_ method: String, _ args: [String: Any]) throws -> Any {
    try decodeAny(LlamaCore.call(method, try jsonString(args)))
  }

  /// A unique temp dir for each test; removed on teardown.
  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-phase4-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir {
      try? FileManager.default.removeItem(at: tempDir)
    }
  }

  private func copyScript(_ name: String, to sub: String) throws -> String {
    let from = Self.fixturesDir.appendingPathComponent("engine").appendingPathComponent(name)
    let to = tempDir.appendingPathComponent(sub)
    try FileManager.default.createDirectory(
      at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: from, to: to)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: to.path)
    return to.path
  }

  private func writeExec(at path: String, _ content: String) throws {
    try FileManager.default.createDirectory(
      at: URL(fileURLWithPath: path).deletingLastPathComponent(),
      withIntermediateDirectories: true)
    try content.write(toFile: path, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
  }

  // MARK: - managedPath / metalConfig / URLs

  func testManagedPathAndUrlsThroughFFI() throws {
    let f = try loadJSON("engine.json")

    for c in try XCTUnwrap(f["managedPath"] as? [[String: Any]]) {
      let out = try call("engine.managedPath", ["home": c["home"] ?? ""])
      XCTAssertEqual(out["path"] as? String, c["path"] as? String)
    }

    for c in try XCTUnwrap(f["metalConfig"] as? [[String: Any]]) {
      let out = try call("engine.metalConfig", ["brand": c["brand"] ?? ""])
      if let config = c["config"] as? String, config != "unsupportedHardware" {
        XCTAssertEqual(out["config"] as? String, config, "brand \(c["brand"] ?? "")")
      } else {
        XCTAssertEqual(out["error"] as? String, "unsupportedHardware", "brand \(c["brand"] ?? "")")
      }
    }

    for c in try XCTUnwrap(f["artifactUrls"] as? [[String: Any]]) {
      let out = try call("engine.artifactUrl", [
        "version": c["version"] ?? "",
        "config": c["config"] ?? "",
      ])
      XCTAssertEqual(out["url"] as? String, c["url"] as? String)
    }

    for c in try XCTUnwrap(f["unzstdUrls"] as? [[String: Any]]) {
      let out = try call("engine.unzstdUrl", ["version": c["version"] ?? ""])
      XCTAssertEqual(out["url"] as? String, c["url"] as? String)
    }
  }

  // MARK: - binariesResolve (builds the fixture's trees in a temp dir)

  func testBinariesResolveThroughFFI() throws {
    let f = try loadJSON("engine.json")
    let cases = try XCTUnwrap(f["resolveCases"] as? [[String: Any]])

    for (i, c) in cases.enumerated() {
      let caseRoot = tempDir.appendingPathComponent("case-\(i)").path
      let name = c["name"] as? String ?? "case-\(i)"
      let home = (c["home"] as! String).replacingOccurrences(of: "<ROOT>", with: caseRoot)
      let managed = c["managed"] as? [String: Any] ?? [:]
      let managedFile = caseRoot + "/home/.llama-app/llama"

      if (managed["exists"] as? Bool) ?? false {
        try writeExec(
          at: managedFile, managed["exec"] as? Bool == false ? "managed-non-exec" : "managed")
        if (managed["exec"] as? Bool) == false {
          try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: managedFile)
        }
      }

      var annotatedDirs: [String] = []
      for spec in try XCTUnwrap(c["unmanaged"] as? [[String: Any]]) {
        let dir = (spec["dir"] as! String).replacingOccurrences(of: "<ROOT>", with: caseRoot)
        annotatedDirs.append(dir)
        switch spec["kind"] as? String {
        case "external":
          try writeExec(at: dir + "/llama", "external")
        case "brew":
          let target = dir + "/../Cellar/llama.cpp/1.0.0/bin/llama"
          try writeExec(at: target, "brew")
          try FileManager.default.createSymbolicLink(
            atPath: dir + "/llama", withDestinationPath: "../Cellar/llama.cpp/1.0.0/bin/llama")
        default:
          break
        }
      }

      let raw = try callAny("engine.binariesResolve", [
        "home": home,
        "unmanagedDirs": annotatedDirs,
        "ignoreUnmanaged": (c["ignoreUnmanaged"] as? Bool) ?? false,
      ])

      let expected = c["expected"] as? [String: Any] ?? [:]
      let expectedOrigin = expected["origin"] as? String ?? "none"
      if expectedOrigin == "none" {
        XCTAssertTrue(raw is NSNull, "\(name): expected no resolution, got \(raw)")
      } else {
        guard let out = raw as? [String: Any] else {
          XCTFail("\(name): expected a resolution, got \(raw)")
          continue
        }
        XCTAssertEqual(out["origin"] as? String, expectedOrigin, "\(name) origin")
        let expectedPath = ((expected["path"] as? String) ?? "")
          .replacingOccurrences(of: "<ROOT>", with: caseRoot)
        XCTAssertEqual(out["path"] as? String, expectedPath, "\(name) path")
      }
    }
  }

  // MARK: - readVersion + installer leaves (fake subprocesses)

  func testReadVersionThroughFFI() throws {
    let ok = try copyScript("fake-llama-ok", to: "bin/llama")
    let stderr = try copyScript("fake-llama-stderr", to: "stderr/llama")
    let failing = try copyScript("fake-llama-fail", to: "fail/llama")

    var out = try call("engine.readVersion", ["path": ok])
    XCTAssertEqual(out["build"] as? Int64, 11429)
    out = try call("engine.readVersion", ["path": stderr])
    XCTAssertEqual(out["build"] as? Int64, 11429, "version on stderr must be found")
    out = try call("engine.readVersion", ["path": failing])
    XCTAssertEqual(out["build"] as? NSNull, NSNull(), "non-zero exit → no version")
    out = try call("engine.readVersion", ["path": tempDir.appendingPathComponent("missing").path])
    XCTAssertEqual(out["build"] as? NSNull, NSNull())
  }

  func testDecompressThroughFFI() throws {
    let okUnzstd = try copyScript("fake-unzstd", to: "tools/unzstd")
    let badUnzstd = try copyScript("fake-unzstd-fail", to: "tools/unzstd-bad")
    let zst = tempDir.appendingPathComponent("llama-app.zst").path
    try "payload".write(toFile: zst, atomically: true, encoding: .utf8)

    let dest = tempDir.appendingPathComponent("out/llama").path
    var out = try call("engine.decompress", [
      "unzstd": okUnzstd, "zst": zst, "dest": dest,
    ])
    XCTAssertEqual(out["ok"] as? Bool, true)
    XCTAssertEqual(try String(contentsOfFile: dest, encoding: .utf8), "payload")
    let mode = (try FileManager.default.attributesOfItem(atPath: dest)[.posixPermissions]) as! Int
    XCTAssertEqual(mode & 0o111, 0o111, "decompressed binary must be executable")

    let destBadURL = tempDir.appendingPathComponent("out2").appendingPathComponent("llama")
    let destBad = destBadURL.path
    out = try call("engine.decompress", [
      "unzstd": badUnzstd, "zst": zst, "dest": destBad,
    ])
    XCTAssertEqual(out["error"] as? String, "decompressFailed")
    XCTAssertEqual(
      try FileManager.default.contentsOfDirectory(
        atPath: destBadURL.deletingLastPathComponent().path).count,
      0, "temp file must be cleaned up")
  }

  func testAtomicInstallThroughFFI() throws {
    let src = tempDir.appendingPathComponent("staged/llama").path
    try writeExec(at: src, "staged-bytes")

    let dest = tempDir.appendingPathComponent("deep/nested/llama").path
    var out = try call("engine.atomicInstall", ["src": src, "dest": dest])
    XCTAssertEqual(out["ok"] as? Bool, true)
    XCTAssertEqual(try String(contentsOfFile: dest, encoding: .utf8), "staged-bytes")
    let mode = (try FileManager.default.attributesOfItem(atPath: dest)[.posixPermissions]) as! Int
    XCTAssertEqual(mode & 0o111, 0o111)

    out = try call("engine.atomicInstall", [
      "src": tempDir.appendingPathComponent("nope").path, "dest": dest,
    ])
    XCTAssertEqual(out["error"] as? String, "installFailed")
  }

  func testEnsurePathSymlinkThroughFFI() throws {
    let home = tempDir.appendingPathComponent("home").path
    let live = tempDir.appendingPathComponent("live/llama").path
    try writeExec(at: live, "x")

    // Fresh: creates ~/.local/bin and the symlink.
    var out = try call("engine.ensurePathSymlink", ["home": home, "llamaPath": live])
    XCTAssertEqual(out["outcome"] as? String, "linked")

    let link = home + "/.local/bin/llama"
    XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: link), live)

    // A real file is left alone.
    try FileManager.default.removeItem(atPath: link)
    try "user-file".write(toFile: link, atomically: true, encoding: .utf8)
    out = try call("engine.ensurePathSymlink", ["home": home, "llamaPath": live])
    XCTAssertEqual(out["outcome"] as? String, "leftExisting")
    XCTAssertEqual(try String(contentsOfFile: link, encoding: .utf8), "user-file")
  }

  func testPromoteStagedThroughFFI() throws {
    let home = tempDir.appendingPathComponent("home").path
    let live = tempDir.appendingPathComponent("live/llama").path

    // Nothing staged.
    var out = try call("engine.promoteStaged", [
      "staged": tempDir.appendingPathComponent("missing").path,
      "live": live, "targetBuild": 11429, "home": home,
    ])
    XCTAssertEqual(out["outcome"] as? String, "nothingStaged")

    // Staged but wrong version → discarded.
    let old = try copyScript("fake-llama-ok", to: "staged-old/llama")
    out = try call("engine.promoteStaged", [
      "staged": old, "live": live, "targetBuild": 9831, "home": home,
    ])
    XCTAssertEqual(out["outcome"] as? String, "discardedStale")
    XCTAssertFalse(FileManager.default.fileExists(atPath: old))
    XCTAssertFalse(FileManager.default.fileExists(atPath: live))

    // Matching target → promoted + PATH symlink refreshed.
    let staged = try copyScript("fake-llama-ok", to: "staged/llama")
    out = try call("engine.promoteStaged", [
      "staged": staged, "live": live, "targetBuild": 11429, "home": home,
    ])
    XCTAssertEqual(out["outcome"] as? String, "promoted")
    let mode = (try FileManager.default.attributesOfItem(atPath: live)[.posixPermissions]) as! Int
    XCTAssertEqual(mode & 0o111, 0o111)
    XCTAssertEqual(
      try FileManager.default.destinationOfSymbolicLink(atPath: home + "/.local/bin/llama"), live)
  }

  // MARK: - fit / parseFitp / memory cache

  func testFitAndParseThroughFFI() throws {
    let f = try loadJSON("engine.json")

    for c in try XCTUnwrap(f["fitParams"] as? [[String: Any]]) {
      let raw = try callAny("engine.fit", [
        "loMib": c["loMib"] ?? 0,
        "hiMib": c["hiMib"] ?? 0,
        "ctxLo": c["ctxLo"] ?? 0,
        "ctxHi": c["ctxHi"] ?? 0,
      ])
      if let expected = c["profile"] as? [String: Any] {
        let out = try XCTUnwrap(raw as? [String: Any])
        let profile = try XCTUnwrap(out["profile"] as? [String: Any])
        XCTAssertEqual(
          profile["ctxBytesPer1kTokens"] as? Int64,
          (expected["ctxBytesPer1kTokens"] as? NSNumber)?.int64Value)
        XCTAssertEqual(
          profile["residentBytes"] as? Int64,
          (expected["residentBytes"] as? NSNumber)?.int64Value)
        XCTAssertEqual(profile["schemaVersion"] as? Int64, 2)
      } else {
        XCTAssertTrue(raw is NSNull, "expected no profile, got \(raw)")
      }
    }

    for c in try XCTUnwrap(f["fitpParse"] as? [[String: Any]]) {
      let out = try call("engine.parseFitp", ["output": c["output"] ?? ""])
      XCTAssertEqual(
        out["sum"] as? Int64, (c["sum"] as? NSNumber)?.int64Value,
        "parse \(c["name"] ?? "")")
    }
  }

  func testMemCacheThroughFFI() throws {
    let f = try loadJSON("engine.json")
    let cacheDir = tempDir.appendingPathComponent("Caches").path
    let model = "org/Model:Q4_0"

    for c in try XCTUnwrap(f["memProfileHash"] as? [[String: Any]]) {
      let out = try call("engine.memCacheFileName", ["modelId": c["modelId"] ?? ""])
      XCTAssertEqual(out["hash"] as? String, c["hash"] as? String)
    }

    var out = try call("engine.memCacheGet", ["cacheDir": cacheDir, "modelId": model])
    XCTAssertEqual(out["profile"] as? NSNull, NSNull(), "fresh dir → miss")

    out = try call("engine.memCacheSet", [
      "cacheDir": cacheDir, "modelId": model,
      "ctxBytesPer1kTokens": 62_000, "residentBytes": 505_413_632,
    ])
    XCTAssertEqual(out["ok"] as? Bool, true)

    out = try call("engine.memCacheGet", ["cacheDir": cacheDir, "modelId": model])
    let profile = try XCTUnwrap(out["profile"] as? [String: Any])
    XCTAssertEqual(profile["ctxBytesPer1kTokens"] as? Int64, 62_000)
    XCTAssertEqual(profile["residentBytes"] as? Int64, 505_413_632)
  }

  /// The Rust cache reads/writes the exact JSON bytes the Swift `Codable`
  /// original produces, so <0.4-style caches survive the migration.
  func testMemCacheJSONInteropsWithSwiftCodable() throws {
    let cacheDir = tempDir.appendingPathComponent("CachesInterop").path
    let model = "unsloth/Tiny-GGUF:Q4_K_M"
    let hash = (try call("engine.memCacheFileName", ["modelId": model]))["hash"] as! String
    let cacheFile = URL(fileURLWithPath: cacheDir).appendingPathComponent("\(hash).json")

    // Swift original writes → Rust reads.
    try FileManager.default.createDirectory(
      at: cacheFile.deletingLastPathComponent(), withIntermediateDirectories: true)
    let swiftProfile = MemProfile(ctxBytesPer1kTokens: 62_000, residentBytes: 505_413_632)
    let data = try JSONEncoder().encode(swiftProfile)
    try data.write(to: cacheFile, options: .atomic)

    let out = try call("engine.memCacheGet", ["cacheDir": cacheDir, "modelId": model])
    let profile = try XCTUnwrap(out["profile"] as? [String: Any])
    XCTAssertEqual(profile["ctxBytesPer1kTokens"] as? Int64, 62_000, "Rust reads Swift bytes")
    XCTAssertEqual(profile["schemaVersion"] as? Int64, 2)

    // Rust writes → Swift original reads (the actual production read path).
    try FileManager.default.removeItem(at: cacheFile)
    _ = try call("engine.memCacheSet", [
      "cacheDir": cacheDir, "modelId": model,
      "ctxBytesPer1kTokens": 7_000, "residentBytes": 1_024,
    ])
    let decoded = try JSONDecoder().decode(MemProfile.self, from: Data(contentsOf: cacheFile))
    XCTAssertEqual(decoded.ctxBytesPer1kTokens, 7_000, "Swift reads Rust bytes")
    XCTAssertEqual(decoded.residentBytes, 1_024)
    XCTAssertEqual(decoded.schemaVersion, 2)
  }
}