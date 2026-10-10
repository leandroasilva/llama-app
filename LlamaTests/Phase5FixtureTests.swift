import CryptoKit
import Foundation
import XCTest

@testable import Llama

/// Fase 5 equivalence: Download engine leaves (`ModelManager+Transfer`,
/// `ActiveDownload`, `DownloadError`) and the `.partial` writer, against the
/// shared `Fixtures/download.json` corpus.
///
/// Parity note: the streaming loop and the per-task `PartialWriter` stay in
/// Swift (`URLSession` delegate + `WriterTable`). The pinned SHA-256 vectors and
/// the crypto-vs-crypto check (`CryptoKit` here, `sha2` in Rust) prove the
/// incremental hash both sides produce is the same one-pass digest, and the
/// writer lifecycle exercises resume re-hash + 200-restart through the FFI.
final class Phase5FixtureTests: XCTestCase {

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

  private func call(_ method: String, _ args: [String: Any]) throws -> [String: Any] {
    try decode(LlamaCore.call(method, try jsonString(args)))
  }

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-phase5-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir {
      try? FileManager.default.removeItem(at: tempDir)
    }
  }

  private func hex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
  }

  // MARK: - Response leaves

  func testFullSizeAndUserMessageThroughFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["fullSizeCases"] as? [[String: Any]]) {
      let out = try call("download.extractFullSize", [
        "status": (c["status"] as? NSNumber)?.int64Value ?? 0,
        "headers": c["headers"] ?? [],
      ])
      XCTAssertEqual(
        out["size"] as? Int64, (c["size"] as? NSNumber)?.int64Value,
        "size case \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["userMessageCases"] as? [[String: Any]]) {
      let out = try call("download.userMessage", ["status": (c["status"] as? NSNumber)?.int64Value ?? 0])
      XCTAssertEqual(out["message"] as? String, c["message"] as? String, "message \(c["name"] ?? "")")
    }
  }

  func testTransientRetryDecisionsThroughFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["transientCodes"] as? [[String: Any]]) {
      let out = try call("download.isTransient", ["code": (c["code"] as? NSNumber)?.int64Value ?? 0])
      XCTAssertEqual(out["transient"] as? Bool, c["transient"] as? Bool, "code \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["retryDelays"] as? [[String: Any]]) {
      let out = try call("download.retryDelay", [
        "attempts": (c["attempts"] as? NSNumber)?.int64Value ?? 0,
        "base": (c["base"] as? NSNumber)?.doubleValue ?? 2.0,
      ])
      let delay = (out["delay"] as? NSNumber)?.doubleValue ?? -1
      let expected = (c["delay"] as? NSNumber)?.doubleValue ?? -1
      XCTAssertEqual(delay, expected, accuracy: 1e-9, "retry \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["transientDecisions"] as? [[String: Any]]) {
      let out = try call("download.transientDecision", [
        "isNetworkAvailable": (c["isNetworkAvailable"] as? Bool) ?? true,
        "attempts": (c["attempts"] as? NSNumber)?.int64Value ?? 0,
        "maxAttempts": (c["maxAttempts"] as? NSNumber)?.int64Value ?? 3,
      ])
      XCTAssertEqual(out["outcome"] as? String, c["outcome"] as? String, "decision \(c["name"] ?? "")")
      XCTAssertEqual(
        out["resumeOnReconnect"] as? Bool, c["resumeOnReconnect"] as? Bool,
        "decision \(c["name"] ?? "")")
    }
  }

  func testRateAndProgressThroughFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["rateCases"] as? [[String: Any]]) {
      let out = try call("download.rate", [
        "samples": c["samples"] ?? [],
        "window": (c["window"] as? NSNumber)?.doubleValue ?? 5.0,
      ])
      if let expected = (c["rate"] as? NSNumber)?.int64Value {
        XCTAssertEqual(out["rate"] as? Int64, expected, "rate \(c["name"] ?? "")")
      } else {
        XCTAssertEqual(out["rate"] as? NSNull, NSNull(), "rate \(c["name"] ?? "") → null")
      }
    }

    for c in try XCTUnwrap(f["progressCases"] as? [[String: Any]]) {
      let out = try call("download.refreshProgress", [
        "totalUnitCount": (c["totalUnitCount"] as? NSNumber)?.int64Value ?? 0,
        "completedFilesBytes": (c["completedFilesBytes"] as? NSNumber)?.int64Value ?? 0,
        "activeBytes": (c["activeBytes"] as? NSNumber)?.int64Value ?? 0,
        "expectedActiveBytes": (c["expectedActiveBytes"] as? NSNumber)?.int64Value ?? 0,
      ])
      XCTAssertEqual(
        out["totalUnitCount"] as? Int64, (c["totalUnitCountAfter"] as? NSNumber)?.int64Value,
        "progress total \(c["name"] ?? "")")
      XCTAssertEqual(
        out["completedUnitCount"] as? Int64,
        (c["completedUnitCountAfter"] as? NSNumber)?.int64Value,
        "progress completed \(c["name"] ?? "")")
    }
  }

  func testErrorMessagesThroughFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["errorMessageCases"] as? [[String: Any]]) {
      var args: [String: Any] = ["kind": c["kind"] ?? ""]
      if let reason = c["reason"] { args["reason"] = reason }
      if let required = c["required"] { args["required"] = required }
      if let available = c["available"] { args["available"] = available }
      let out = try call("download.errorMessage", args)
      XCTAssertEqual(out["description"] as? String, c["description"] as? String, "error \(c["name"] ?? "")")
      XCTAssertEqual(out["recovery"] as? String, c["recovery"] as? String, "error \(c["name"] ?? "")")
    }
  }

  func testHashVerifyAndTooSmallThroughFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["hashVerifyCases"] as? [[String: Any]]) {
      let out = try call("download.verifyHash", [
        "computed": c["computed"] ?? "", "expected": c["expected"] ?? "",
      ])
      XCTAssertEqual(out["matches"] as? Bool, c["matches"] as? Bool, "verify \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["tooSmallCases"] as? [[String: Any]]) {
      let out = try call("download.isTooSmall", ["size": (c["size"] as? NSNumber)?.int64Value ?? 0])
      XCTAssertEqual(out["tooSmall"] as? Bool, c["tooSmall"] as? Bool, "too small \(c["name"] ?? "")")
    }
  }

  // MARK: - Hashing: Rust `sha2` == Swift `CryptoKit` on the pinned vectors

  func testSha256VectorsMatchCryptoKitAndFFI() throws {
    let f = try loadJSON("download.json")

    for c in try XCTUnwrap(f["sha256Vectors"] as? [[String: Any]]) {
      let input = c["input"] as? String ?? ""
      let expected = c["hex"] as? String ?? ""

      // Swift's own hasher agrees with the pinned vector…
      XCTAssertEqual(hex(SHA256.hash(data: Data(input.utf8))), expected, "vector \(input)")

      // …and so does the Rust core over the same bytes.
      let out = try call("partial.sha256", ["text": input])
      XCTAssertEqual(out["hex"] as? String, expected, "ffi vector \(input)")
    }
  }

  // MARK: - Partial writer lifecycle (resume re-hash, 200 restart)

  func testPartialWriterThroughFFI() throws {
    let partial = tempDir.appendingPathComponent("model.gguf.partial").path

    // Open fresh: zero bytes, empty digest.
    var out = try call("partial.open", ["path": partial])
    let handle = (out["handle"] as! NSNumber).int64Value
    XCTAssertEqual(out["bytesWritten"] as? Int64, 0)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data())))

    // Streamed chunks accumulate into the one-pass digest.
    out = try call("partial.append", ["handle": handle, "text": "abc"])
    XCTAssertEqual(out["bytesWritten"] as? Int64, 3)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data("abc".utf8))))

    out = try call("partial.append", ["handle": handle, "text": "def"])
    XCTAssertEqual(out["bytesWritten"] as? Int64, 6)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data("abcdef".utf8))))

    // The on-disk file re-hashes to the same digest (the resume-time path).
    out = try call("partial.feedHash", ["path": partial])
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data("abcdef".utf8))))

    // 200 restart: truncate + reset hash + counter.
    out = try call("partial.restart", ["handle": handle])
    XCTAssertEqual(out["ok"] as? Bool, true)
    XCTAssertEqual(out["bytesWritten"] as? Int64, 0)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data())))

    // Reopen over a pre-seeded prefix re-hashes it and continues from there.
    try "prefix-resume-".write(toFile: partial, atomically: true, encoding: .utf8)
    out = try call("partial.open", ["path": partial])
    let resumedHandle = (out["handle"] as! NSNumber).int64Value
    XCTAssertEqual(out["bytesWritten"] as? Int64, 14)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data("prefix-resume-".utf8))))

    out = try call("partial.append", ["handle": resumedHandle, "text": "tail"])
    XCTAssertEqual(out["bytesWritten"] as? Int64, 18)
    XCTAssertEqual(out["digest"] as? String, hex(SHA256.hash(data: Data("prefix-resume-tail".utf8))))

    // Unknown handle is classified, not a crash.
    out = try call("partial.append", ["handle": 424_242, "text": "x"])
    XCTAssertEqual(out["error"] as? String, "notFound")

    // Close drops the registry entries.
    XCTAssertEqual((try call("partial.close", ["handle": resumedHandle]))["ok"] as? Bool, true)
    XCTAssertEqual((try call("partial.close", ["handle": handle]))["ok"] as? Bool, true)
  }

  func testFeedHashMissingPathIsClassifiedError() throws {
    let out = try call("partial.feedHash", [
      "path": tempDir.appendingPathComponent("nope").path,
    ])
    XCTAssertEqual(out["error"] as? String, "openFailed")
    XCTAssertNotNil(out["message"])
  }
}