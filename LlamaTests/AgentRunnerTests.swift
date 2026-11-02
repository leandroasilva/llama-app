import XCTest

@testable import Llama

final class AgentRunnerTests: XCTestCase {
  private var root: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-agent-runner-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  func testApprovedWriteRunsAndItsResultIsFedBack() async throws {
    let recorder = Recorder()
    let runner = makeRunner(recorder) { callNumber in
      if callNumber == 1 {
        return .completed(
          content: "", toolCalls: [self.writeCall(path: "hello.txt", content: "hi")],
          finishReason: "tool_calls")
      }
      return .completed(content: "Done", toolCalls: [], finishReason: "stop")
    }

    var events: [AgentEvent] = []
    try await runner.send(
      "Create hello.txt",
      onEvent: { events.append($0) },
      approval: { _ in .allowAlways })

    let contents = try String(
      contentsOf: root.appendingPathComponent("hello.txt"), encoding: .utf8)
    XCTAssertEqual(contents, "hi")
    XCTAssertTrue(events.contains(.assistantCompleted("Done")))
    XCTAssertTrue(events.contains { if case .toolResult(_, _, false) = $0 { return true } else { return false } })
    XCTAssertEqual(recorder.messages.count, 2)

    let toolMessage = recorder.messages.last?.last
    XCTAssertEqual(toolMessage?["role"] as? String, "tool")
    XCTAssertTrue((toolMessage?["content"] as? String)?.contains("Created hello.txt") == true)
  }

  func testDeniedWriteIsReportedBackAndKeepsTheFileUntouched() async throws {
    let recorder = Recorder()
    let runner = makeRunner(recorder) { callNumber in
      if callNumber == 1 {
        return .completed(
          content: "", toolCalls: [self.writeCall(path: "hello.txt", content: "hi")],
          finishReason: "tool_calls")
      }
      return .completed(content: "Okay", toolCalls: [], finishReason: "stop")
    }

    var events: [AgentEvent] = []
    try await runner.send(
      "Create hello.txt",
      onEvent: { events.append($0) },
      approval: { _ in .deny })

    XCTAssertFalse(
      FileManager.default.fileExists(atPath: root.appendingPathComponent("hello.txt").path))
    XCTAssertTrue(events.contains { if case .toolResult(_, _, true) = $0 { return true } else { return false } })

    let toolMessage = recorder.messages.last?.last
    XCTAssertEqual(toolMessage?["role"] as? String, "tool")
    XCTAssertTrue((toolMessage?["content"] as? String)?.contains("denied") == true)
  }

  func testAutoApproveSkipsTheApprovalHandler() async throws {
    let recorder = Recorder()
    let runner = makeRunner(recorder) { callNumber in
      if callNumber == 1 {
        return .completed(
          content: "", toolCalls: [self.writeCall(path: "auto.txt", content: "ok")],
          finishReason: "tool_calls")
      }
      return .completed(content: "Done", toolCalls: [], finishReason: "stop")
    }
    runner.autoApprove = true

    try await runner.send("Create auto.txt")

    let contents = try String(
      contentsOf: root.appendingPathComponent("auto.txt"), encoding: .utf8)
    XCTAssertEqual(contents, "ok")
  }

  func testTextAndReasoningDeltasAreForwarded() async throws {
    let recorder = Recorder()
    let runner = AgentRunner(
      modelID: "test-model", workspaceURL: root,
      streamFactory: { _, messages, _ in
        recorder.messages.append(messages)
        return AsyncThrowingStream { continuation in
          continuation.yield(.reasoningDelta("thinking"))
          continuation.yield(.textDelta("Hel"))
          continuation.yield(.textDelta("lo"))
          continuation.yield(.completed(content: "Hello", toolCalls: [], finishReason: "stop"))
          continuation.finish()
        }
      })

    var events: [AgentEvent] = []
    try await runner.send("Hi", onEvent: { events.append($0) })

    XCTAssertTrue(events.contains(.reasoningDelta("thinking")))
    XCTAssertTrue(events.contains(.textDelta("Hel")))
    XCTAssertTrue(events.contains(.assistantCompleted("Hello")))
  }

  // MARK: - Helpers

  private func makeRunner(
    _ recorder: Recorder,
    turn: @escaping (Int) -> AgentStreamEvent
  ) -> AgentRunner {
    AgentRunner(modelID: "test-model", workspaceURL: root) { _, messages, _ in
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

  private final class Recorder {
    var messages: [[[String: Any]]] = []
  }
}
