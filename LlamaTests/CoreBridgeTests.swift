import Foundation
import XCTest

@testable import Llama

/// Fase 7 bridge checks: the `CoreConfig` snapshot and core-event channel over
/// the FFI, pinned to the same `Fixtures/core_config.json` and
/// `Fixtures/core_events.json` the Rust suite consumes.
final class CoreBridgeTests: XCTestCase {

  private func load(_ name: String) throws -> [String: Any] {
    let url = Fixtures.directory.appendingPathComponent(name)
    let data = try Data(contentsOf: url)
    return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
  }

  func testConfigFixtureRoundTripsThroughCore() throws {
    let fixture = try load("core_config.json")
    let cases = try XCTUnwrap(fixture["cases"] as? [[String: Any]])
    XCTAssertFalse(cases.isEmpty)

    for c in cases {
      let name = c["name"] as? String ?? "?"
      let input = c["input"] as? [String: Any] ?? [:]
      let expected = try XCTUnwrap(c["normalized"] as? [String: Any])

      LlamaCore.setConfig(input)
      let config = try XCTUnwrap(LlamaCore.snapshot()?["config"] as? [String: Any])
      XCTAssertEqual(config as NSDictionary, expected as NSDictionary, "core_config case \(name)")
    }
  }

  func testSnapshotShapeMatchesFixture() throws {
    let fixture = try load("core_config.json")
    let shape = Set(try XCTUnwrap(fixture["snapshotShape"] as? [String]))
    let snapshot = try XCTUnwrap(LlamaCore.snapshot())
    let config = try XCTUnwrap(snapshot["config"] as? [String: Any])
    XCTAssertEqual(Set(config.keys), shape)
    XCTAssertEqual((snapshot["schemaVersion"] as? NSNumber)?.intValue, 1)
  }

  func testEventMappingMatchesFixture() throws {
    let fixture = try load("core_events.json")
    let events = try XCTUnwrap(fixture["events"] as? [[String: Any]])
    XCTAssertFalse(events.isEmpty)
    for e in events {
      let kind = try XCTUnwrap(e["kind"] as? String)
      let notification = try XCTUnwrap(e["notification"] as? String)
      XCTAssertEqual(
        EventBridge.notification(forKind: kind)?.rawValue, notification, "mapping for \(kind)")
    }
  }

  func testUnknownEventKindHasNoNotification() {
    XCTAssertNil(EventBridge.notification(forKind: "mystery"))
  }

  func testEventCallbackRePostsNotification() {
    EventBridge.start()
    let posted = expectation(description: "serverState re-posted as a notification")
    let token = NotificationCenter.default.addObserver(
      forName: .LBServerStateDidChange, object: nil, queue: nil
    ) { note in
      if (note.userInfo?["state"] as? String) == "running" { posted.fulfill() }
    }
    defer {
      NotificationCenter.default.removeObserver(token)
      LlamaCore.installEventCallback(nil)
    }

    let response = LlamaCore.call(
      "event.emit", #"{"kind":"serverState","payload":{"state":"running"}}"#)
    XCTAssertEqual(response, #"{"known":true,"ok":true}"#)
    wait(for: [posted], timeout: 2)
  }
}
