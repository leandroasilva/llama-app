import Foundation
import XCTest

@testable import Llama

final class LlamaURLTests: XCTestCase {

  /// Tests run inside the dev build, which registers `llama-dev`, so pin the
  /// production scheme explicitly instead of depending on `Bundle.main`.
  private let schemes: Set<String> = ["llama", "llamabarn"]

  private func parse(_ s: String) -> LlamaURL? {
    guard let url = URL(string: s) else { return nil }
    return LlamaURL.parse(url, schemes: schemes)
  }

  func testParsesInstallWithRepoAndQuant() {
    XCTAssertEqual(
      parse("llama://install?repo=org/model-GGUF&quant=Q4_K_M"),
      .install(repo: "org/model-GGUF", quant: "Q4_K_M"))
  }

  func testCollapsesEmptyQuantToNil() {
    XCTAssertEqual(
      parse("llama://install?repo=org/model"), .install(repo: "org/model", quant: nil))
    XCTAssertEqual(
      parse("llama://install?repo=org/model&quant="), .install(repo: "org/model", quant: nil))
  }

  func testRejectsUnregisteredScheme() {
    XCTAssertNil(parse("https://install?repo=org/model"))
    XCTAssertNil(parse("evil://install?repo=org/model"))
  }

  func testRejectsUnknownVerb() {
    XCTAssertNil(parse("llama://uninstall?repo=org/model"))
  }

  func testRejectsRepoThatIsNotExactlyTwoNonEmptyParts() {
    XCTAssertNil(parse("llama://install?repo=org"))
    XCTAssertNil(parse("llama://install?repo=org/model/extra"))
    XCTAssertNil(parse("llama://install?repo=/model"))
    XCTAssertNil(parse("llama://install?repo=org/"))
  }

  func testRejectsRepoThatCouldRewriteTheRequestURL() {
    // Percent-encoded `?`/`#`/`/` decode into the repo value, which is then
    // interpolated into the HF API URL. Reject them before that can happen.
    XCTAssertNil(parse("llama://install?repo=org/mo%3Fdel"))
    XCTAssertNil(parse("llama://install?repo=org/mo%23del"))
    XCTAssertNil(parse("llama://install?repo=org/mo%2Fdel"))
    XCTAssertNil(parse("llama://install?repo=org/mo%20del"))
  }

  func testAcceptsRepoCharsHuggingFaceAllows() {
    XCTAssertEqual(
      parse("llama://install?repo=The-Bloke/Meta-Llama-3.1-8B-Instruct.Q4_K_M"),
      .install(repo: "The-Bloke/Meta-Llama-3.1-8B-Instruct.Q4_K_M", quant: nil))
  }
}
