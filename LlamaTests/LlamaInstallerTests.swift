import Foundation
import XCTest

@testable import Llama

final class LlamaInstallerTests: XCTestCase {

  func testDigestMatchesAcceptsTheAdvertisedHash() {
    let hash = String(repeating: "a", count: 64)
    XCTAssertTrue(LlamaInstaller.digestMatches(computed: hash, expected: hash))
    // HF serves lowercase hex, but the comparison shouldn't depend on it.
    XCTAssertTrue(LlamaInstaller.digestMatches(computed: hash, expected: hash.uppercased()))
  }

  func testDigestMatchesRejectsADifferentHash() {
    let computed = String(repeating: "a", count: 64)
    let expected = String(repeating: "b", count: 64)
    XCTAssertFalse(LlamaInstaller.digestMatches(computed: computed, expected: expected))
  }

  func testDigestMatchesTrustsTheBytesWhenNoHashIsAdvertised() {
    // No advertised hash (e.g. a non-LFS file): nothing to check against.
    XCTAssertTrue(LlamaInstaller.digestMatches(computed: "anything", expected: nil))
  }
}
