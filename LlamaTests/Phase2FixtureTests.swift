import Foundation
import XCTest

@testable import Llama

/// Fase 2 equivalence: filesystem I/O (`GGUFMetadata`, `HFCache`) against the
/// shared `Fixtures/*.json` corpus. The Swift originals and the Rust port both
/// consume these fixtures; this test asserts Swift == fixture (and spot-checks
/// the Rust FFI where it exposes the same call).
final class Phase2FixtureTests: XCTestCase {

  // MARK: - Fixture loading

  private static let fixturesDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures")

  private var cacheDir: URL {
    Self.fixturesDir.appendingPathComponent("hf_cache/cache")
  }

  private func load<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
    let url = Self.fixturesDir.appendingPathComponent(name)
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
  }

  // MARK: - GGUFMetadata

  private struct GGUFMetadataFixture: Decodable {
    struct ParsedInt: Decodable { let parsed: Bool; let length: Int? }
    struct ParsedBool: Decodable { let parsed: Bool; let value: Bool? }
    struct FileCase: Decodable {
      let file: String
      let contextLength: ParsedInt
      let hasEmbeddedMTPHead: ParsedBool
      let isDecisionModel: ParsedBool
    }
    let files: [FileCase]
  }

  func testGGUFMetadataFixtures() throws {
    let f = try load("gguf_metadata.json", as: GGUFMetadataFixture.self)
    for c in f.files {
      let path = Self.fixturesDir.appendingPathComponent(c.file).path

      let ctx = GGUFMetadata.contextLength(path: path)
      if c.contextLength.parsed {
        XCTAssertNotNil(ctx, "contextLength(\(c.file)) should parse")
        XCTAssertEqual(ctx!, c.contextLength.length, "contextLength(\(c.file))")
      } else {
        XCTAssertNil(ctx, "contextLength(\(c.file)) should be unparseable")
      }

      let mtp = GGUFMetadata.hasEmbeddedMTPHead(path: path)
      if c.hasEmbeddedMTPHead.parsed {
        XCTAssertEqual(mtp, c.hasEmbeddedMTPHead.value, "hasEmbeddedMTPHead(\(c.file))")
      } else {
        XCTAssertNil(mtp, "hasEmbeddedMTPHead(\(c.file)) should be unparseable")
      }

      let dec = GGUFMetadata.isDecisionModel(path: path)
      if c.isDecisionModel.parsed {
        XCTAssertEqual(dec, c.isDecisionModel.value, "isDecisionModel(\(c.file))")
      } else {
        XCTAssertNil(dec, "isDecisionModel(\(c.file)) should be unparseable")
      }

      // Spot-check the Rust FFI: it collapses the outer "unparseable" state to
      // a bare `null` response and otherwise reports `{parsed, ...}`.
      let args = try jsonString(["path": path])
      let raw = LlamaCore.call("gguf.contextLength", args)
      if c.contextLength.parsed {
        let v = try XCTUnwrap(try decode(raw))
        XCTAssertEqual(v["parsed"] as? Bool, true)
        XCTAssertEqual(v["length"] as? Int, c.contextLength.length, "ffi length(\(c.file))")
      } else {
        XCTAssertEqual(raw, "null", "ffi contextLength(\(c.file))")
      }
    }
  }

  // MARK: - HFCache

  private struct PathHelpersFixture: Decodable {
    struct NameCase: Decodable { let url: String; let name: String? }
    struct PathCase: Decodable { let url: String; let path: String? }
    let repoDirName: [NameCase]
    let repoRelativePath: [PathCase]
  }

  private struct ScanEntry: Decodable {
    let id: String
    let ctxWindow: Int
    let fileSize: Int64
    let draftHead: String?
    let isDecisionModel: Bool
    let usesMTP: Bool
    let mmprojFile: String?
    let draftSidecarFile: String?
    let additionalParts: [String]
    let hfRepoDirName: String
    let modelFile: String
  }

  private struct ScanFixture: Decodable { let cacheDir: String; let entries: [ScanEntry] }

  private struct PartialBytesCase: Decodable { let modelId: String; let bytes: Int64 }

  private struct ScanPlaceholdersFixture: Decodable { let cacheDir: String; let paths: [String] }

  private struct SnapshotExistsCase: Decodable {
    let repoDir: String; let filename: String; let value: Bool
  }

  private struct WriteBlobFixture: Decodable {
    let repoDir: String
    let commit: String
    let blobHash: String
    let filename: String
    let expectedSymlinkTarget: String
    let expectedRefsMain: String
  }

  private struct DeleteExpected: Decodable {
    let modelFileExists: Bool
    let mmprojFileExists: Bool
    let blobsDirExists: Bool
    let snapshotsDirExists: Bool
    let repoDirExists: Bool
    let refsMainExists: Bool
  }

  private struct DeleteFixture: Decodable {
    let repoDir: String
    let modelFile: String
    let additionalParts: [String]
    let mmprojFile: String?
    let expected: DeleteExpected
  }

  private struct HFCacheFixture: Decodable {
    let pathHelpers: PathHelpersFixture
    let scan: ScanFixture
    let partialBytes: [PartialBytesCase]
    let scanPlaceholders: ScanPlaceholdersFixture
    let snapshotFileExists: [SnapshotExistsCase]
    let writeBlobAndLink: WriteBlobFixture
    let deleteModelFiles: DeleteFixture
  }

  private func hfFixture() throws -> HFCacheFixture {
    try load("hf_cache.json", as: HFCacheFixture.self)
  }

  func testHFCachePathHelpers() throws {
    let f = try hfFixture()
    for c in f.pathHelpers.repoDirName {
      XCTAssertEqual(
        HFCache.repoDirName(from: URL(string: c.url)!), c.name, "repoDirName(\(c.url))")
    }
    for c in f.pathHelpers.repoRelativePath {
      XCTAssertEqual(
        HFCache.repoRelativePath(from: URL(string: c.url)!), c.path, "repoRelativePath(\(c.url))")
    }
  }

  func testHFCacheScanForSideloaded() throws {
    let f = try hfFixture()
    let cache = Self.fixturesDir.appendingPathComponent(f.scan.cacheDir)
    let results = HFCache.scanForSideloaded(cacheDir: cache)

    var byId: [String: (entry: Model, paths: ResolvedPaths)] = [:]
    for r in results { byId[r.entry.id] = r }
    XCTAssertEqual(byId.count, f.scan.entries.count, "entry count")

    for c in f.scan.entries {
      let got = try XCTUnwrap(byId[c.id], "missing entry \(c.id)")
      XCTAssertEqual(got.entry.ctxWindow, c.ctxWindow, "\(c.id) ctx")
      XCTAssertEqual(got.entry.fileSize, c.fileSize, "\(c.id) size")
      XCTAssertEqual(got.entry.scannedDraftHead?.rawValue, c.draftHead, "\(c.id) draftHead")
      XCTAssertEqual(got.entry.isDecisionModel, c.isDecisionModel, "\(c.id) decision")
      XCTAssertEqual(got.paths.usesMTP, c.usesMTP, "\(c.id) usesMTP")
      XCTAssertEqual(got.paths.hfRepoDirName, c.hfRepoDirName, "\(c.id) hfRepoDirName")
      XCTAssertEqual(got.paths.modelFile, cache.appendingPathComponent(c.modelFile).path)
      XCTAssertEqual(
        got.paths.mmprojFile,
        c.mmprojFile.map { cache.appendingPathComponent($0).path }, "\(c.id) mmproj")
      XCTAssertEqual(
        got.paths.draftSidecarFile,
        c.draftSidecarFile.map { cache.appendingPathComponent($0).path }, "\(c.id) sidecar")
      XCTAssertEqual(
        got.paths.additionalParts,
        c.additionalParts.map { cache.appendingPathComponent($0).path }, "\(c.id) parts")
    }
  }

  func testHFCachePartialBytes() throws {
    let f = try hfFixture()
    for c in f.partialBytes {
      XCTAssertEqual(
        HFCache.partialBytes(cacheDir: cacheDir, modelId: c.modelId), c.bytes,
        "partialBytes(\(c.modelId))")
    }
  }

  func testHFCacheScanPlaceholders() throws {
    let f = try hfFixture()
    let cache = Self.fixturesDir.appendingPathComponent(f.scanPlaceholders.cacheDir)
    let got = HFCache.scanPlaceholders(cacheDir: cache)
      .map { $0.path.replacingOccurrences(of: cache.path + "/", with: "") }
      .sorted()
    XCTAssertEqual(got, f.scanPlaceholders.paths.sorted())
  }

  func testHFCacheSnapshotFileExists() throws {
    let f = try hfFixture()
    for c in f.snapshotFileExists {
      XCTAssertEqual(
        HFCache.snapshotFileExists(cacheDir: cacheDir, repoDir: c.repoDir, filename: c.filename),
        c.value, "\(c.repoDir)/\(c.filename)")
    }
  }

  func testHFCacheWriteBlobAndLink() throws {
    let f = try hfFixture()
    let w = f.writeBlobAndLink
    let tmp = try makeTempDir(tag: "wbl")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let cache = tmp.appendingPathComponent("cache")
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)

    let tempFile = tmp.appendingPathComponent("upload.tmp")
    try Data("hello-blob".utf8).write(to: tempFile)

    try HFCache.writeBlobAndLink(
      cacheDir: cache, repoDir: w.repoDir, commit: w.commit, blobHash: w.blobHash,
      filename: w.filename, from: tempFile)

    let blob = cache.appendingPathComponent(w.repoDir).appendingPathComponent("blobs")
      .appendingPathComponent(w.blobHash)
    XCTAssertEqual(try Data(contentsOf: blob), Data("hello-blob".utf8))
    XCTAssertFalse(FileManager.default.fileExists(atPath: tempFile.path), "temp moved")

    let symlink = cache.appendingPathComponent(w.repoDir).appendingPathComponent("snapshots")
      .appendingPathComponent(w.commit).appendingPathComponent(w.filename)
    XCTAssertEqual(
      try FileManager.default.destinationOfSymbolicLink(atPath: symlink.path),
      w.expectedSymlinkTarget)

    let refsMain = cache.appendingPathComponent(w.repoDir).appendingPathComponent("refs/main")
    XCTAssertEqual(try String(contentsOf: refsMain, encoding: .utf8), w.expectedRefsMain)
  }

  func testHFCacheDeleteModelFiles() throws {
    let f = try hfFixture()
    let d = f.deleteModelFiles
    let tmp = try makeTempDir(tag: "delete")
    defer { try? FileManager.default.removeItem(at: tmp) }
    let cache = tmp.appendingPathComponent("cache")
    try copyTree(from: cacheDir, to: cache)

    let paths = ResolvedPaths(
      modelFile: cache.appendingPathComponent(d.modelFile).path,
      additionalParts: d.additionalParts.map { cache.appendingPathComponent($0).path },
      mmprojFile: d.mmprojFile.map { cache.appendingPathComponent($0).path },
      hfRepoDirName: d.repoDir)

    try HFCache.deleteModelFiles(cacheDir: cache, repoDir: d.repoDir, paths: paths)

    let fm = FileManager.default
    let repo = cache.appendingPathComponent(d.repoDir)
    XCTAssertEqual(
      fm.fileExists(atPath: cache.appendingPathComponent(d.modelFile).path), d.expected.modelFileExists)
    let mmprojExists = d.mmprojFile.map {
      fm.fileExists(atPath: cache.appendingPathComponent($0).path)
    } ?? false
    XCTAssertEqual(mmprojExists, d.expected.mmprojFileExists)
    XCTAssertEqual(fm.fileExists(atPath: repo.appendingPathComponent("blobs").path), d.expected.blobsDirExists)
    XCTAssertEqual(
      fm.fileExists(atPath: repo.appendingPathComponent("snapshots").path), d.expected.snapshotsDirExists)
    XCTAssertEqual(fm.fileExists(atPath: repo.path), d.expected.repoDirExists)
    XCTAssertEqual(
      fm.fileExists(atPath: repo.appendingPathComponent("refs/main").path), d.expected.refsMainExists)
  }

  // MARK: - Helpers

  private func jsonString(_ object: [String: Any]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: object)
    return String(decoding: data, as: UTF8.self)
  }

  private func decode(_ raw: String?) throws -> [String: Any]? {
    let raw = try XCTUnwrap(raw)
    return try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
  }

  private func makeTempDir(tag: String) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-phase2-\(tag)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  /// Copies a directory tree preserving symlinks (as symlinks) so the mutating
  /// HFCache cases can operate on a throwaway copy instead of the fixture tree.
  private func copyTree(from src: URL, to dst: URL) throws {
    let fm = FileManager.default
    try fm.createDirectory(at: dst, withIntermediateDirectories: true)
    for entry in try fm.contentsOfDirectory(
      at: src, includingPropertiesForKeys: [.isSymbolicLinkKey, .isDirectoryKey])
    {
      let to = dst.appendingPathComponent(entry.lastPathComponent)
      let values = try entry.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
      if values.isSymbolicLink == true {
        try fm.createSymbolicLink(
          atPath: to.path, withDestinationPath: try fm.destinationOfSymbolicLink(atPath: entry.path))
      } else if values.isDirectory == true {
        try copyTree(from: entry, to: to)
      } else {
        try fm.copyItem(at: entry, to: to)
      }
    }
  }
}
