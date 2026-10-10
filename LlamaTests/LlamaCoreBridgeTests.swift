import Foundation
import XCTest

@testable import Llama

/// End-to-end check of the FFI: Swift calls through `LlamaCore` into the Rust
/// `llama-core-ffi` staticlib, whose JSON must match the results the pure-Swift
/// implementations produce for the same inputs.
final class LlamaCoreBridgeTests: XCTestCase {

  func testPing() {
    XCTAssertEqual(LlamaCore.call("ping"), #"{"pong":true}"#)
  }

  func testVersionParseThroughRust() throws {
    let json = try XCTUnwrap(LlamaCore.call("version.parse", #"{"output":"b9370-aa50b2c2a"}"#))
    XCTAssertEqual(json, #"{"build":9370}"#)

    let none = try XCTUnwrap(LlamaCore.call("version.parse", #"{"output":"garbage"}"#))
    XCTAssertEqual(none, #"{"build":null}"#)
  }

  func testRepoParseThroughRust() throws {
    let json = try XCTUnwrap(
      LlamaCore.call("repo.parse", #"{"repoDir":"models--org--repo--extra"}"#))
    XCTAssertEqual(json, #"{"org":"org","repo":"repo--extra"}"#)
  }

  func testCoreAgreesWithSwiftOnCorpus() throws {
    let url = Fixtures.directory.appendingPathComponent("llama_version.json")
    struct Fixture: Decodable {
      struct Case: Decodable {
        let input: String
        let build: Int?
      }
      let cases: [Case]
    }
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))

    for c in fixture.cases {
      let args = try JSONEncoder().encode(["output": c.input])
      let json = try XCTUnwrap(LlamaCore.call("version.parse", String(decoding: args, as: UTF8.self)))
      XCTAssertEqual(json, #"{"build":\#(c.build.map(String.init) ?? "null")}"#,
                     "Rust disagrees for \(c.input.debugDescription)")
    }
  }
}
