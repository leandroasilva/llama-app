import Foundation
import XCTest

@testable import Llama

/// Fase 6 equivalence: `LlamaServer` / `LlamaServerAPI` pure leaves and the
/// `reclaimPort` / interface-classification helpers, against the shared
/// `Fixtures/server.json` corpus.
///
/// Parity note: the syscalls themselves stay in Swift (`getifaddrs`, ARP
/// `sysctl`, `Process`/`URLSession`, `lsof`/`ps`/`kill`, the port bind probe) and
/// `webuiUrl` stays Swift because of `URLComponents` percent-encoding. The Rust
/// leaves receive the syscall outputs as inputs, so this suite pins the decision
/// logic on both sides of the FFI. Cases that read the filesystem carry a
/// `<TMP>` placeholder that both suites substitute with their own temp dir.
final class Phase6FixtureTests: XCTestCase {

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

  /// Replaces the fixture `<TMP>` token with this test's scratch dir.
  private func subst(_ value: Any, _ tmp: String) -> Any {
    if let s = value as? String { return s.replacingOccurrences(of: "<TMP>", with: tmp) }
    if let a = value as? [Any] { return a.map { subst($0, tmp) } }
    if let d = value as? [String: Any] { return d.mapValues { subst($0, tmp) } }
    return value
  }

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-phase6-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
  }

  // MARK: - Quoting + rendered invocation

  func testQuoteAndDisplayCommandThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["quoteCases"] as? [[String: Any]]) {
      let out = try call("server.quote", ["text": c["text"] ?? ""])
      XCTAssertEqual(out["quoted"] as? String, c["quoted"] as? String, "quote \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["displayCommandCases"] as? [[String: Any]]) {
      let out = try call("server.displayCommand", [
        "executablePath": c["executablePath"] ?? "",
        "arguments": c["arguments"] ?? [],
        "extraArgCount": (c["extraArgCount"] as? NSNumber)?.intValue ?? 0,
        "env": c["env"] ?? [],
      ])
      XCTAssertEqual(out["command"] as? String, c["command"] as? String, "display \(c["name"] ?? "")")
    }
  }

  // MARK: - Host argument + Tailscale/interface classification

  func testHostArgumentThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["hostArgumentCases"] as? [[String: Any]]) {
      var args: [String: Any] = ["bindAddress": c["bindAddress"] ?? ""]
      if let build = c["versionBuild"], !(build is NSNull) { args["versionBuild"] = build }
      let out = try call("server.hostArgument", args)
      XCTAssertEqual(out["host"] as? String, c["expected"] as? String, "host \(c["name"] ?? "")")
    }
  }

  func testTailscaleClassificationThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["tailscaleCases"] as? [[String: Any]]) {
      let out = try call("server.isTailscaleAddress", ["address": c["ip"] ?? ""])
      XCTAssertEqual(
        out["isTailscale"] as? Bool, c["expected"] as? Bool, "tailscale \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["tailscaleAddressCases"] as? [[String: Any]]) {
      let out = try call("server.tailscaleAddress", ["addresses": c["addresses"] ?? []])
      assertOptionalString(out["address"], c["expected"], "tailscaleAddr \(c["name"] ?? "")")
    }
  }

  func testIpPickAndHostsThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["pickIpCases"] as? [[String: Any]]) {
      var args: [String: Any] = ["addresses": c["addresses"] ?? []]
      if let primary = c["primary"], !(primary is NSNull) { args["primary"] = primary }
      let out = try call("server.pickIpAddress", args)
      assertOptionalString(out["address"], c["expected"], "pickIp \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["localHostCases"] as? [[String: Any]]) {
      var args: [String: Any] = [:]
      if let bind = c["effectiveBind"], !(bind is NSNull) { args["effectiveBind"] = bind }
      let out = try call("server.localHost", args)
      XCTAssertEqual(out["host"] as? String, c["expected"] as? String, "localHost \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["resolvedHostCases"] as? [[String: Any]]) {
      var args: [String: Any] = ["addresses": c["addresses"] ?? []]
      if let bind = c["effectiveBind"], !(bind is NSNull) { args["effectiveBind"] = bind }
      if let primary = c["primary"], !(primary is NSNull) { args["primary"] = primary }
      let out = try call("server.resolvedHost", args)
      XCTAssertEqual(out["host"] as? String, c["expected"] as? String, "resolvedHost \(c["name"] ?? "")")
    }
  }

  // MARK: - Bind/webui fallbacks + fit target

  func testEffectiveBindWebuiAndFitThroughFFI() throws {
    let f = try loadJSON("server.json")
    let tmp = tempDir.path

    for c in try XCTUnwrap(f["effectiveBindCases"] as? [[String: Any]]) {
      var args: [String: Any] = ["localAddrs": c["localAddrs"] ?? []]
      if let configured = c["configured"], !(configured is NSNull) { args["configured"] = configured }
      let out = try call("server.effectiveBindAddress", args)
      assertOptionalString(out["address"], c["expected"], "effectiveBind \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["effectiveWebuiCases"] as? [[String: Any]]) {
      let dir = (try XCTUnwrap(c["dir"] as? String)).replacingOccurrences(of: "<TMP>", with: tmp)
      try FileManager.default.createDirectory(
        atPath: dir, withIntermediateDirectories: true)
      if (c["indexed"] as? Bool) == true {
        try "<html></html>".write(toFile: (dir as NSString).appendingPathComponent("index.html"),
                                  atomically: true, encoding: .utf8)
      }
      let out = try call("server.effectiveWebuiDirectory", ["configured": dir])
      let expected = c["expected"] as? String == nil ? nil : dir
      assertOptionalString(out["directory"], expected, "effectiveWebui \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["fitTargetCases"] as? [[String: Any]]) {
      let out = try call("server.fitTargetMb", [
        "gpuWorkingSetMb": (c["gpuWorkingSetMb"] as? NSNumber)?.int64Value ?? 0,
        "memoryMb": (c["memoryMb"] as? NSNumber)?.int64Value ?? 0,
        "useFullWorkingSet": (c["useFullWorkingSet"] as? Bool) ?? false,
      ])
      XCTAssertEqual(
        out["mb"] as? Int64, (c["expected"] as? NSNumber)?.int64Value, "fit \(c["name"] ?? "")")
    }
  }

  // MARK: - Preset rejection + model statuses

  func testPresetRejectionAndModelStatusesThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["presetRejectionCases"] as? [[String: Any]]) {
      let out = try call("server.presetRejectionOption", ["message": c["message"] ?? ""])
      assertOptionalString(out["option"], c["expected"], "preset \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["modelsStatusCases"] as? [[String: Any]]) {
      let out = try call("server.parseModelStatuses", ["body": c["body"] ?? ""])
      if let expected = c["statuses"] as? [[String]] {
        let actual = try XCTUnwrap(out["statuses"] as? [[String]], "statuses \(c["name"] ?? "")")
        XCTAssertEqual(actual, expected, "statuses \(c["name"] ?? "")")
      } else {
        XCTAssertEqual(out["error"] as? String, "malformed", "status error \(c["name"] ?? "")")
      }
    }
  }

  // MARK: - Reclaim port leaves

  func testReclaimPortThroughFFI() throws {
    let f = try loadJSON("server.json")

    for c in try XCTUnwrap(f["reclaimCases"] as? [[String: Any]]) {
      if let lsof = c["lsofOutput"] {
        let out = try call("server.extractPids", ["lsofOutput": lsof])
        let actual = (out["pids"] as? [NSNumber])?.map(\.int64Value) ?? []
        let expected = (c["pids"] as? [NSNumber])?.map(\.int64Value) ?? []
        XCTAssertEqual(actual, expected, "pids \(c["name"] ?? "")")
      }
      if let comms = c["comms"] as? [[String: Any]] {
        for comm in comms {
          let out = try call("server.isLlamaProcess", ["comm": comm["comm"] ?? ""])
          XCTAssertEqual(
            out["isLlama"] as? Bool, comm["isLlama"] as? Bool, "comm \(c["name"] ?? "")")
        }
      }
      if let paths = c["paths"] as? [[String: Any]] {
        for path in paths {
          let out = try call("server.lastPathComponent", ["path": path["path"] ?? ""])
          XCTAssertEqual(
            out["name"] as? String, path["name"] as? String, "path \(c["name"] ?? "")")
        }
      }
    }
  }

  // MARK: - Full launch spec tree

  func testBuildLaunchSpecThroughFFI() throws {
    let f = try loadJSON("server.json")
    let tmp = tempDir.path
    let caseKeys = [
      "llamaPath", "presetsPath", "logFilePath", "hfCacheDirectory",
      "emptyCacheDirectory", "configuredBindAddress", "configuredWebUiDirectory",
      "apiKey",
    ]

    for raw in try XCTUnwrap(f["buildLaunchSpecCases"] as? [[String: Any]]) {
      let c = subst(raw, tmp) as! [String: Any]

      // Stage the custom web UI folder the case points at.
      if let dir = c["configuredWebUiDirectory"] as? String {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try "<html></html>".write(
          toFile: (dir as NSString).appendingPathComponent("index.html"),
          atomically: true, encoding: .utf8)
      }

      var args: [String: Any] = [:]
      for key in caseKeys where !(c[key] is NSNull) { args[key] = c[key] }
      args["addresses"] = c["addresses"] ?? []
      args["port"] = (c["port"] as? NSNumber)?.int64Value ?? 0
      args["fitTargetMb"] = (c["fitTargetMb"] as? NSNumber)?.int64Value ?? 0
      args["extraArgs"] = c["extraArgs"] ?? []
      args["agentMode"] = (c["agentMode"] as? Bool) ?? false
      if let primary = c["primaryInterface"], !(primary is NSNull) { args["primaryInterface"] = primary }
      if let build = c["versionBuild"], !(build is NSNull) { args["currentVersionBuild"] = build }
      if let sleep = c["sleepIdleSeconds"], !(sleep is NSNull) { args["sleepIdleSeconds"] = sleep }

      let out = try call("server.buildLaunchSpec", args)
      if let expected = c["spec"] as? [String: Any] {
        let spec = try XCTUnwrap(out["spec"] as? [String: Any], "spec \(c["name"] ?? "")")
        XCTAssertEqual(
          spec["executablePath"] as? String, expected["executablePath"] as? String,
          "exe \(c["name"] ?? "")")
        XCTAssertEqual(
          spec["arguments"] as? [String], expected["arguments"] as? [String],
          "args \(c["name"] ?? "")")
        XCTAssertEqual(
          (spec["extraArgCount"] as? NSNumber)?.intValue,
          (expected["extraArgCount"] as? NSNumber)?.intValue, "extra \(c["name"] ?? "")")
        let actualEnv = (spec["env"] as? [[String]]) ?? []
        let expectedEnv = (expected["env"] as? [[String]]) ?? []
        XCTAssertEqual(actualEnv, expectedEnv, "env \(c["name"] ?? "")")
      } else {
        XCTAssertEqual(out["error"] as? String, "noBinary", "noBinary \(c["name"] ?? "")")
      }
    }
  }

  // MARK: - Small assertion helper for nullable string fields

  private func assertOptionalString(
    _ actual: Any?, _ expected: Any?, _ message: String,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    if let expected = expected as? String {
      XCTAssertEqual(actual as? String, expected, message, file: file, line: line)
    } else {
      XCTAssertTrue(actual is NSNull, "\(message) → expected null, got \(String(describing: actual))",
                    file: file, line: line)
    }
  }
}
