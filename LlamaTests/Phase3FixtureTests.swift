import Foundation
import XCTest

@testable import Llama

/// Fase 3 equivalence: network + remote selection against the shared
/// `Fixtures/*.json` corpus.
///
/// Parity note: `HFRepoResolver.resolve(_:quant:)` and `Catalog.picks/parseBytes`
/// are `private static` (file-scoped) in the Swift originals, so they can't be
/// invoked from `@testable`. The pure logic therefore lives in the Rust port and
/// is pinned here by the shared fixtures (Rust `phase3.rs` asserts the same
/// numbers on its side). `HFRequest.make` *is* testable, and is exercised
/// directly below against the same fixture semantics the Rust `hfRequest`
/// module implements.
final class Phase3FixtureTests: XCTestCase {

  // MARK: - Fixture loading

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

  // MARK: - HFRepoResolver (through FFI)

  func testHFRepoResolverFixturesThroughFFI() throws {
    let f = try loadJSON("hf_repo_resolver.json")
    let cases = try XCTUnwrap(f["cases"] as? [[String: Any]])
    for c in cases {
      guard let name = c["name"] as? String else { continue }
      let out = try call("net.resolveFromSiblings", [
        "repo": c["repo"] ?? "",
        "requestedQuant": c["requestedQuant"] as Any,
        "budgetMb": c["budgetMb"] ?? 0,
        "siblings": c["siblings"] ?? [],
      ])

      if let expected = c["expected"] as? [String: Any] {
        XCTAssertEqual(out["error"] as? String, nil, "\(name): expected success, got error")
        XCTAssertEqual(out["modelId"] as? String, expected["modelId"] as? String, "\(name) modelId")
        XCTAssertEqual(out["mainUrl"] as? String, expected["mainUrl"] as? String, "\(name) mainUrl")
        XCTAssertEqual(
          out["additionalParts"] as? [String], expected["additionalParts"] as? [String],
          "\(name) parts")
        XCTAssertEqual(out["mmprojUrl"] as? String, expected["mmprojUrl"] as? String, "\(name) mmproj")
        XCTAssertEqual(out["draftUrl"] as? String, expected["draftUrl"] as? String, "\(name) draft")
        XCTAssertEqual(
          out["approximateBytes"] as? Int64, (expected["approximateBytes"] as? NSNumber)?.int64Value,
          "\(name) bytes")
      } else if let expectedError = c["expectedError"] as? String {
        XCTAssertEqual(out["error"] as? String, expectedError, "\(name) error kind")
        if let shard = c["errorShard"] as? String {
          let message = try XCTUnwrap(out["message"] as? String, "\(name) shard message")
          XCTAssertTrue(message.contains(shard), "\(name) message mentions \(shard)")
        }
      } else {
        XCTFail("\(name): fixture must set expected or expectedError")
      }
    }
  }

  // MARK: - Catalog (through FFI)

  func testCatalogParseBytesThroughFFI() throws {
    let f = try loadJSON("catalog.json")
    let fixture = try XCTUnwrap(f["parseBytes"] as? [[String: Any]])
    for c in fixture {
      var args: [String: Any] = [:]
      if let label = c["label"] as? String { args["label"] = label }
      let out = try call("catalog.parseBytes", args)
      XCTAssertEqual(
        out["bytes"] as? Int64, (c["bytes"] as? NSNumber)?.int64Value,
        "parseBytes(\(String(describing: c["label"])))")
    }
  }

  func testCatalogFeaturedSuggestionsThroughFFI() throws {
    let f = try loadJSON("catalog.json")
    let cases = try XCTUnwrap(f["cases"] as? [[String: Any]])
    for c in cases {
      guard let name = c["name"] as? String else { continue }
      let out = try decode(LlamaCore.call("catalog.featuredSuggestions", try jsonString([
        "families": f["families"] ?? [],
        "budgetMb": c["budgetMb"] ?? 0,
        "systemMemoryMb": c["systemMemoryMb"] ?? 0,
      ])))
      let got = try XCTUnwrap(out["suggestions"] as? [[String: Any]])
      let expected = try XCTUnwrap(c["expected"] as? [[String: Any]])

      XCTAssertEqual(got.count, expected.count, "\(name) suggestion count")
      for (g, e) in zip(got, expected) {
        XCTAssertEqual(g["brand"] as? String, e["brand"] as? String, "\(name) brand")
        XCTAssertEqual(g["repo"] as? String, e["repo"] as? String, "\(name) repo")
        XCTAssertEqual(g["quant"] as? String, e["quant"] as? String, "\(name) quant")
        XCTAssertEqual(g["sizeLabel"] as? String, e["sizeLabel"] as? String, "\(name) sizeLabel")
        XCTAssertEqual(g["hasVision"] as? Bool, e["hasVision"] as? Bool, "\(name) hasVision")
      }
    }
  }

  // MARK: - HFRequest (direct Swift vs fixture)

  /// The fixture's request semantics, exercised on the *Swift* original so the
  /// Rust port's `hfRequest` module is checked against both the fixture and the
  /// code it replaces.
  /// `HFRequest.make` now delegates the header decision to the core; this pins
  /// the Swift wrapper's translation (and the `AppInfo.userAgent` shape).
  func testRequestMakeDelegatesToCore() throws {
    let f = try loadJSON("hf_http.json")
    let cases = try XCTUnwrap(f["requests"] as? [[String: Any]])

    for c in cases {
      let expected = try XCTUnwrap(c["expected"] as? [[String]])
      let url = try XCTUnwrap(URL(string: c["url"] as! String))
      let request = HFRequest.make(url, token: c["token"] as? String)
      // `AppInfo.userAgent` is runtime-derived (bundle + OS), so the literal is
      // pinned by the FFI test (injectable) and the format check below; here we
      // only assert the presence/absence semantics the fixture encodes.
      for pair in expected {
        let want = pair[0] == "User-Agent" ? AppInfo.userAgent : pair[1]
        XCTAssertEqual(
          request.value(forHTTPHeaderField: pair[0]), want,
          "header \(pair[0]) for \(c["name"] ?? "")")
      }
      XCTAssertEqual(
        request.value(forHTTPHeaderField: "Authorization"),
        expected.first(where: { $0[0] == "Authorization" })?[1],
        "Authorization for \(c["name"] ?? "")")
    }

    // The app's runtime user agent must match the shape the Rust `hfRequest`
    // module produces for the same inputs (the fixture's pinned examples).
    let ua = AppInfo.userAgent
    let pattern = #"^llama-app/[\d.]+ \(macOS \d+\.\d+; arm64\)$"#
    XCTAssertNotNil(
      ua.range(of: pattern, options: .regularExpression),
      "AppInfo.userAgent format: \(ua)")
  }

  // MARK: - HFRequest + response (through FFI)

  func testHFHttpFixturesThroughFFI() throws {
    let f = try loadJSON("hf_http.json")
    let ua = try XCTUnwrap(f["testUserAgent"] as? String)
    let sha = try XCTUnwrap(f["sha256"] as? String)

    for c in try XCTUnwrap(f["userAgent"] as? [[String: Any]]) {
      let out = try call("hfRequest.userAgent", [
        "shortVersion": c["shortVersion"] ?? "",
        "osMajor": c["osMajor"] ?? 0,
        "osMinor": c["osMinor"] ?? 0,
      ])
      XCTAssertEqual(out["ua"] as? String, c["expected"] as? String)
    }

    for c in try XCTUnwrap(f["requests"] as? [[String: Any]]) {
      var args: [String: Any] = [
        "url": c["url"] ?? "",
        "userAgent": ua,
      ]
      if let token = c["token"] as? String { args["token"] = token }
      let out = try call("hfRequest.make", args)
      let headers = try XCTUnwrap(out["headers"] as? [[String]])
      var dict: [String: String] = [:]
      for h in headers { dict[h[0]] = h[1] }
      let expected = try XCTUnwrap(c["expected"] as? [[String]])
      var expectedDict: [String: String] = [:]
      for pair in expected { expectedDict[pair[0]] = pair[1] }
      XCTAssertEqual(dict, expectedDict, "request \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["blobHash"] as? [[String: Any]]) {
      var headers: [[String]] = []
      if let v = c["xLinkedEtag"] as? String { headers.append(["X-Linked-Etag", v]) }
      if let v = c["etag"] as? String {
        headers.append(["ETag", v.replacingOccurrences(of: "<64hex>", with: sha)])
      }
      let out = try call("net.blobHash", ["status": 200, "headers": headers])
      XCTAssertEqual(out["hash"] as? String, c["expected"] as? String, "blobHash \(c["name"] ?? "")")
    }

    for c in try XCTUnwrap(f["commitHash"] as? [[String: Any]]) {
      var headers: [[String]] = []
      if let v = c["xRepoCommit"] as? String { headers.append(["X-Repo-Commit", v]) }
      let out = try call("net.commitHash", ["status": 200, "headers": headers])
      XCTAssertEqual(out["hash"] as? String, c["expected"] as? String)
    }

    for c in try XCTUnwrap(f["redirect"] as? [[String: Any]]) {
      var args: [String: Any] = [:]
      if let from = c["from"] as? String { args["from"] = from }
      if let to = c["to"] as? String { args["to"] = to }
      let out = try call("net.sameHostRedirect", args)
      XCTAssertEqual(out["value"] as? Bool, c["expected"] as? Bool)
    }

    for c in try XCTUnwrap(f["hosts"] as? [[String: Any]]) {
      let out = try call("hfRequest.isHuggingfaceHost", ["url": c["url"] ?? ""])
      XCTAssertEqual(
        out["value"] as? Bool, c["expected"] as? Bool, "host \(c["url"] ?? "")")
    }
  }

  /// A redirect to a Hugging Face host keeps the token; a hop to any other host
  /// must not carry it. This is the download path's version of the same rule
  /// `HFRequest.make` applies when the request is first built.
  func testRedirectRequestScopesTokenToHuggingFace() {
    func request(_ url: String) -> URLRequest {
      var r = URLRequest(url: URL(string: url)!)
      r.setValue("Bearer hf_secret", forHTTPHeaderField: "Authorization")
      r.setValue("llama-app/test", forHTTPHeaderField: "User-Agent")
      return r
    }

    let onHF = HFRequest.redirectRequest(request("https://cdn-lfs.huggingface.co/x"))
    XCTAssertEqual(onHF.value(forHTTPHeaderField: "Authorization"), "Bearer hf_secret")
    XCTAssertEqual(onHF.value(forHTTPHeaderField: "User-Agent"), "llama-app/test")

    let offHF = HFRequest.redirectRequest(request("https://cas-bridge.xethub.hf.co/x"))
    XCTAssertNil(offHF.value(forHTTPHeaderField: "Authorization"))
    XCTAssertEqual(offHF.value(forHTTPHeaderField: "User-Agent"), "llama-app/test")
  }
}