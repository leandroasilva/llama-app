import Foundation

enum AgentStreamEvent {
  /// A token (or larger piece) of assistant text.
  case textDelta(String)
  /// A token of the model's reasoning stream, when the model has one.
  case reasoningDelta(String)
  /// The turn finished. `toolCalls` is empty for a normal text answer.
  case completed(content: String, toolCalls: [AgentToolCall], finishReason: String?)
}

enum AgentClientError: LocalizedError {
  case notConfigured
  case http(Int, String)
  case invalidResponse

  var errorDescription: String? {
    switch self {
    case .notConfigured:
      return "The local server isn't ready yet."
    case .http(let status, let body):
      let detail = body.trimmingCharacters(in: .whitespacesAndNewlines)
      return detail.isEmpty
        ? "The local server returned HTTP \(status)."
        : "The local server returned HTTP \(status): \(detail)"
    case .invalidResponse:
      return "The local server sent a response the agent couldn't read."
    }
  }
}

/// Streams one chat completion from llama-server's OpenAI-compatible endpoint,
/// accumulating streamed tool-call fragments into complete calls. llama-server
/// is local, so a plain URLSession is enough; no cloud client is involved.
struct AgentClient {
  let baseURL: URL
  let apiKey: String?
  let session: URLSession

  static let defaultSession: URLSession = {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.urlCache = nil
    configuration.timeoutIntervalForRequest = 600
    configuration.timeoutIntervalForResource = 3600
    return URLSession(configuration: configuration)
  }()

  init(baseURL: URL, apiKey: String?, session: URLSession = AgentClient.defaultSession) {
    self.baseURL = baseURL
    self.apiKey = apiKey
    self.session = session
  }

  func stream(
    model: String, messages: [[String: Any]], tools: [[String: Any]]
  ) -> AsyncThrowingStream<AgentStreamEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          try await run(
            model: model, messages: messages, tools: tools, continuation: continuation)
          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in task.cancel() }
    }
  }

  private func run(
    model: String, messages: [[String: Any]], tools: [[String: Any]],
    continuation: AsyncThrowingStream<AgentStreamEvent, Error>.Continuation
  ) async throws {
    var request = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
    request.httpMethod = "POST"
    request.timeoutInterval = 600
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let body: [String: Any] = [
      "model": model,
      "messages": messages,
      "tools": tools,
      "tool_choice": "auto",
      "stream": true,
      "temperature": 0.2,
    ]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (bytes, response) = try await session.bytes(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw AgentClientError.invalidResponse
    }
    guard (200..<300).contains(http.statusCode) else {
      var detail = ""
      for try await line in bytes.lines {
        detail += line
        if detail.count > 4000 { break }
      }
      throw AgentClientError.http(http.statusCode, detail)
    }

    var content = ""
    var reasoning = ""
    var finishReason: String?
    var calls: [Int: ToolCallAccumulator] = [:]

    for try await line in bytes.lines {
      if Task.isCancelled { throw CancellationError() }
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      guard trimmed.hasPrefix("data:") else { continue }
      let payload = trimmed.dropFirst("data:".count)
        .trimmingCharacters(in: .whitespaces)
      if payload == "[DONE]" { break }
      guard let data = payload.data(using: .utf8),
        let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data),
        let choice = chunk.choices.first
      else { continue }

      if let delta = choice.delta {
        if let piece = delta.content, !piece.isEmpty {
          content += piece
          continuation.yield(.textDelta(piece))
        }
        if let piece = delta.reasoningContent ?? delta.reasoning, !piece.isEmpty {
          reasoning += piece
          continuation.yield(.reasoningDelta(piece))
        }
        for deltaCall in delta.toolCalls ?? [] {
          let index = deltaCall.index ?? 0
          var accumulator = calls[index] ?? ToolCallAccumulator()
          if let id = deltaCall.id, !id.isEmpty { accumulator.id = id }
          if let name = deltaCall.function?.name, !name.isEmpty {
            accumulator.name = name
          }
          if let fragment = deltaCall.function?.arguments {
            accumulator.arguments += fragment
          }
          calls[index] = accumulator
        }
      }

      if let finish = choice.finishReason {
        finishReason = finish
      }
    }

    let completedCalls = calls.keys.sorted().compactMap { index -> AgentToolCall? in
      guard let accumulator = calls[index], !accumulator.name.isEmpty else { return nil }
      return AgentToolCall(
        id: accumulator.id ?? "call_\(index)_\(UUID().uuidString.prefix(8))",
        name: accumulator.name,
        arguments: accumulator.arguments.isEmpty ? "{}" : accumulator.arguments)
    }

    continuation.yield(
      .completed(content: content, toolCalls: completedCalls, finishReason: finishReason))
    _ = reasoning
  }

  private struct ToolCallAccumulator {
    var id: String?
    var name = ""
    var arguments = ""
  }

  private struct StreamChunk: Decodable {
    let choices: [Choice]

    struct Choice: Decodable {
      let delta: Delta?
      let finishReason: String?

      enum CodingKeys: String, CodingKey {
        case delta
        case finishReason = "finish_reason"
      }
    }

    struct Delta: Decodable {
      let content: String?
      let reasoningContent: String?
      let reasoning: String?
      let toolCalls: [ToolCallDelta]?

      enum CodingKeys: String, CodingKey {
        case content
        case reasoningContent = "reasoning_content"
        case reasoning
        case toolCalls = "tool_calls"
      }
    }

    struct ToolCallDelta: Decodable {
      let index: Int?
      let id: String?
      let function: Function?

      struct Function: Decodable {
        let name: String?
        let arguments: String?
      }
    }
  }
}
