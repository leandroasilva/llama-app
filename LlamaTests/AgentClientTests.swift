import XCTest

@testable import Llama

final class AgentClientTests: XCTestCase {
  override func tearDown() {
    MockAgentURLProtocol.statusCode = 200
    MockAgentURLProtocol.chunks = []
    super.tearDown()
  }

  func testStreamsTextReasoningAndToolCallFragments() async throws {
    MockAgentURLProtocol.statusCode = 200
    MockAgentURLProtocol.chunks = [
      sse(["choices": [["delta": ["content": "Hel"]]]]),
      sse(["choices": [["delta": ["content": "lo"]]]]),
      sse(["choices": [["delta": ["reasoning_content": "think"]]]]),
      sse([
        "choices": [
          [
            "delta": [
              "tool_calls": [
                [
                  "index": 0, "id": "call_1",
                  "function": ["name": "read", "arguments": "{\"path\":\"a"],
                ]
              ]
            ]
          ]
        ]
      ]),
      sse([
        "choices": [
          [
            "delta": [
              "tool_calls": [
                ["index": 0, "function": ["arguments": ".swift\"}"]]
              ]
            ]
          ]
        ]
      ]),
      sse(["choices": [["delta": [:], "finish_reason": "tool_calls"]]]),
      "data: [DONE]\n\n",
    ]

    var deltas = ""
    var reasoning = ""
    var finalText = ""
    var calls: [AgentToolCall] = []
    var finish: String?

    for try await event in makeClient().stream(model: "test", messages: [], tools: []) {
      switch event {
      case .textDelta(let piece): deltas += piece
      case .reasoningDelta(let piece): reasoning += piece
      case .completed(let content, let toolCalls, let reason):
        finalText = content
        calls = toolCalls
        finish = reason
      }
    }

    XCTAssertEqual(deltas, "Hello")
    XCTAssertEqual(finalText, "Hello")
    XCTAssertEqual(reasoning, "think")
    XCTAssertEqual(finish, "tool_calls")
    XCTAssertEqual(calls.count, 1)
    XCTAssertEqual(calls.first?.id, "call_1")
    XCTAssertEqual(calls.first?.name, "read")
    XCTAssertEqual(calls.first?.arguments, "{\"path\":\"a.swift\"}")
  }

  func testHTTPErrorIncludesTheResponseBody() async {
    MockAgentURLProtocol.statusCode = 500
    MockAgentURLProtocol.chunks = ["{\"error\":\"boom\"}"]

    do {
      for try await _ in makeClient().stream(model: "test", messages: [], tools: []) {}
      XCTFail("Expected the stream to throw")
    } catch let AgentClientError.http(status, body) {
      XCTAssertEqual(status, 500)
      XCTAssertTrue(body.contains("boom"))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  private func makeClient() -> AgentClient {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockAgentURLProtocol.self]
    let session = URLSession(configuration: configuration)
    return AgentClient(
      baseURL: URL(string: "http://localhost:9931/v1")!, apiKey: nil, session: session)
  }

  private func sse(_ object: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: object)
    return "data: \(String(decoding: data, as: UTF8.self))\n\n"
  }
}

final class MockAgentURLProtocol: URLProtocol {
  static var statusCode = 200
  static var chunks: [String] = []

  override class func canInit(with request: URLRequest) -> Bool { true }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    let response = HTTPURLResponse(
      url: request.url!, statusCode: Self.statusCode, httpVersion: "HTTP/1.1",
      headerFields: ["Content-Type": "text/event-stream"])!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    for chunk in Self.chunks {
      client?.urlProtocol(self, didLoad: Data(chunk.utf8))
    }
    client?.urlProtocolDidFinishLoading(self)
  }

  override func stopLoading() {}
}
