import Foundation
import XCTest

@testable import Llama

final class ExtraServerArgsTests: XCTestCase {

  override func tearDown() {
    UserDefaults.standard.removeObject(forKey: "extraServerArgs")
    super.tearDown()
  }

  func testEmptyByDefault() {
    XCTAssertTrue(UserSettings.extraServerArgList.isEmpty)
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testBasicArgsPassThrough() {
    UserDefaults.standard.set("--api-key secret", forKey: "extraServerArgs")
    XCTAssertEqual(UserSettings.extraServerArgList, ["--api-key", "secret"])
    XCTAssertEqual(UserSettings.sanitizedExtraArgs, ["--api-key", "secret"])
  }

  func testHostFlagIsBlocked() {
    UserDefaults.standard.set("--host 0.0.0.0", forKey: "extraServerArgs")
    XCTAssertEqual(UserSettings.extraServerArgList, ["--host", "0.0.0.0"])
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testPortFlagIsBlocked() {
    UserDefaults.standard.set("--port 8080", forKey: "extraServerArgs")
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testBindFlagIsBlocked() {
    UserDefaults.standard.set("--bind 0.0.0.0", forKey: "extraServerArgs")
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testShortHostFlagIsBlocked() {
    UserDefaults.standard.set("-H 0.0.0.0", forKey: "extraServerArgs")
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testShortPortFlagIsBlocked() {
    UserDefaults.standard.set("-p 8080", forKey: "extraServerArgs")
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testMixedArgsOnlyBlockedOnesRemoved() {
    UserDefaults.standard.set("--api-key secret --host 0.0.0.0 --verbose", forKey: "extraServerArgs")
    XCTAssertEqual(
      UserSettings.sanitizedExtraArgs, ["--api-key", "secret", "--verbose"])
  }

  func testMultipleBlockedFlagsAllRemoved() {
    UserDefaults.standard.set("--host 0.0.0.0 --port 8080 --bind 0.0.0.0", forKey: "extraServerArgs")
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }

  func testBlockedFlagWithoutValueStillSkipsNextToken() {
    UserDefaults.standard.set("--host", forKey: "extraServerArgs")
    XCTAssertEqual(UserSettings.extraServerArgList, ["--host"])
    XCTAssertTrue(UserSettings.sanitizedExtraArgs.isEmpty)
  }
}
