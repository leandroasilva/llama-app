import XCTest

@testable import Llama

@MainActor
final class AgentSessionTests: XCTestCase {
  private var root: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-agent-session-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  func testApprovedWriteRunsAndItsResultIsFedBack() async throws {
    let recorder = Recorder()
    let session = makeSession(recorder) { callNumber in
      if callNumber == 1 {
        return .completed(
          content: "", toolCalls: [self.writeCall(path: "hello.txt", content: "hi")],
          finishReason: "tool_calls")
      }
      return .completed(content: "Done", toolCalls: [], finishReason: "stop")
    }

    session.send("Create hello.txt")
    try await waitUntil { session.pendingApproval != nil }
    XCTAssertEqual(session.pendingApproval?.toolName, "write")

    session.resolveApproval(allow: true, remember: true)
    try await waitUntil { !session.isRunning }

    let contents = try String(
      contentsOf: root.appendingPathComponent("hello.txt"), encoding: .utf8)
    XCTAssertEqual(contents, "hi")
    XCTAssertTrue(session.entries.contains { $0.kind == .assistant && $0.text == "Done" })
    XCTAssertEqual(recorder.messages.count, 2)

    let toolMessage = recorder.messages.last?.last
    XCTAssertEqual(toolMessage?["role"] as? String, "tool")
    XCTAssertTrue((toolMessage?["content"] as? String)?.contains("Created hello.txt") == true)
  }

  func testDeniedWriteIsReportedBackAndKeepsTheFileUntouched() async throws {
    let recorder = Recorder()
    let session = makeSession(recorder) { callNumber in
      if callNumber == 1 {
        return .completed(
          content: "", toolCalls: [self.writeCall(path: "hello.txt", content: "hi")],
          finishReason: "tool_calls")
      }
      return .completed(content: "Okay", toolCalls: [], finishReason: "stop")
    }

    session.send("Create hello.txt")
    try await waitUntil { session.pendingApproval != nil }

    session.resolveApproval(allow: false, remember: false)
    try await waitUntil { !session.isRunning }

    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("hello.txt").path))
    XCTAssertTrue(session.entries.contains { $0.kind == .toolResult && $0.isError })

    let toolMessage = recorder.messages.last?.last
    XCTAssertEqual(toolMessage?["role"] as? String, "tool")
    XCTAssertTrue((toolMessage?["content"] as? String)?.contains("denied") == true)
  }

  // MARK: - Helpers

  private func makeSession(
    _ recorder: Recorder,
    turn: @escaping (Int) -> AgentStreamEvent
  ) -> AgentSession {
    AgentSession(modelID: "test-model", workspaceURL: root) { _, messages, _ in
      recorder.messages.append(messages)
      let callNumber = recorder.messages.count
      return AsyncThrowingStream { continuation in
        continuation.yield(turn(callNumber))
        continuation.finish()
      }
    }
  }

  private func writeCall(path: String, content: String) -> AgentToolCall {
    let arguments = ["path": path, "content": content]
    let data = try! JSONSerialization.data(withJSONObject: arguments)
    return AgentToolCall(
      id: UUID().uuidString, name: "write",
      arguments: String(decoding: data, as: UTF8.self))
  }

  private func waitUntil(_ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(3)
    while !condition() {
      if Date() > deadline {
        XCTFail("Timed out waiting for the agent")
        return
      }
      try await Task.sleep(nanoseconds: 5_000_000)
    }
  }

  private final class Recorder {
    var messages: [[[String: Any]]] = []
  }
}
