import Foundation

/// Events emitted while an `AgentRunner` works through one user turn. The
/// runner owns the conversation and the tool loop; callers only render these
/// events, which keeps the terminal UI and any future front end on the same
/// behaviour.
enum AgentEvent: Equatable {
  case assistantStarted
  case textDelta(String)
  case reasoningDelta(String)
  case assistantCompleted(String)
  case toolCall(name: String, title: String, detail: String)
  case toolResult(name: String, text: String, isError: Bool)
  case notice(String)
}

struct AgentApprovalRequest: Equatable {
  let toolName: String
  let title: String
  let detail: String
}

enum AgentApprovalDecision {
  case allow
  case deny
  case allowAlways
}

enum AgentRunnerError: LocalizedError {
  case busy
  case noWorkspace
  case noModel
  case notConfigured

  var errorDescription: String? {
    switch self {
    case .busy:
      return "The agent is already working on a message."
    case .noWorkspace:
      return "Choose a project folder before sending a task."
    case .noModel:
      return "Choose a model before sending a task."
    case .notConfigured:
      return "The local server isn't ready yet."
    }
  }
}

/// Runs the coding-agent loop: stream a completion, execute any tools the
/// model asks for, feed the results back, and repeat until it answers. This is
/// the shared core behind the `llama-agent` CLI; it has no AppKit or SwiftUI
/// dependencies.
final class AgentRunner {
  typealias StreamFactory =
    (String, [[String: Any]], [[String: Any]]) -> AsyncThrowingStream<AgentStreamEvent, Error>
  typealias EventHandler = (AgentEvent) -> Void
  typealias ApprovalHandler = (AgentApprovalRequest) async -> AgentApprovalDecision

  var modelID: String
  let workspace: AgentWorkspace
  var contextTokens: Int
  var maximumSteps: Int
  var autoApprove: Bool

  private let client: AgentClient?
  private let streamFactory: StreamFactory?
  private var history: [AgentMessage] = []
  private var approvedTools: Set<String> = []
  private var isRunning = false

  private static let maximumHistoryMessages = 24

  init(
    modelID: String,
    workspaceURL: URL,
    baseURL: URL? = nil,
    apiKey: String? = nil,
    contextTokens: Int = 8_192,
    maximumSteps: Int = 40,
    autoApprove: Bool = false,
    streamFactory: StreamFactory? = nil
  ) {
    self.modelID = modelID
    self.workspace = AgentWorkspace(root: workspaceURL)
    self.contextTokens = contextTokens
    self.maximumSteps = maximumSteps
    self.autoApprove = autoApprove
    self.streamFactory = streamFactory
    if streamFactory == nil, let baseURL {
      self.client = AgentClient(baseURL: baseURL, apiKey: apiKey)
    } else {
      self.client = nil
    }
  }

  func reset() {
    guard !isRunning else { return }
    history.removeAll()
    approvedTools.removeAll()
  }

  /// Sends one user message and drives the loop to completion. Events are
  /// delivered synchronously on the caller's task; approvals are awaited so a
  /// front end can pause mid-loop.
  func send(
    _ text: String,
    onEvent: @escaping EventHandler = { _ in },
    approval: ApprovalHandler? = nil
  ) async throws {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    guard !modelID.isEmpty else { throw AgentRunnerError.noModel }
    guard !isRunning else { throw AgentRunnerError.busy }
    guard client != nil || streamFactory != nil else { throw AgentRunnerError.notConfigured }

    isRunning = true
    defer { isRunning = false }

    history.append(AgentMessage(role: .user, content: trimmed))
    pruneHistory()

    for step in 1...maximumSteps {
      if Task.isCancelled { throw CancellationError() }

      onEvent(.assistantStarted)

      let turn: AgentTurn
      do {
        turn = try await streamTurn(onEvent: onEvent)
      } catch is CancellationError {
        return
      }

      let content = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
      if !content.isEmpty {
        onEvent(.assistantCompleted(content))
      }

      if !content.isEmpty || !turn.toolCalls.isEmpty {
        history.append(
          AgentMessage(
            role: .assistant, content: content.isEmpty ? nil : content,
            toolCalls: turn.toolCalls))
      }

      guard !turn.toolCalls.isEmpty else {
        if turn.finishReason == "length" {
          onEvent(.notice("The model hit its output limit. Ask it to continue if the answer is incomplete."))
        }
        if content.isEmpty {
          onEvent(.notice("The model returned no text and made no tool calls."))
        }
        return
      }

      for call in turn.toolCalls {
        if Task.isCancelled { throw CancellationError() }

        if AgentToolRunner.isMutating(call.name) {
          let decision: AgentApprovalDecision
          if autoApprove || approvedTools.contains(call.name) {
            decision = .allow
          } else if let approval {
            let summary = AgentToolRunner.summary(for: call, workspace: workspace.root)
            decision = await approval(
              AgentApprovalRequest(
                toolName: call.name, title: summary.title, detail: summary.detail))
          } else {
            decision = .deny
          }

          if decision == .deny {
            let denial = "The user denied this tool call."
            history.append(
              AgentMessage(role: .tool, content: denial, toolCallID: call.id, name: call.name))
            onEvent(.toolResult(name: call.name, text: denial, isError: true))
            continue
          }
          if decision == .allowAlways {
            approvedTools.insert(call.name)
          }
        }

        let summary = AgentToolRunner.summary(for: call, workspace: workspace.root)
        onEvent(.toolCall(name: call.name, title: summary.title, detail: summary.detail))

        let result = await AgentToolRunner.execute(call, workspace: workspace)
        history.append(
          AgentMessage(role: .tool, content: result.text, toolCallID: call.id, name: call.name))
        onEvent(.toolResult(name: call.name, text: result.text, isError: result.isError))
      }

      pruneHistory()

      if step == maximumSteps {
        onEvent(.notice("Stopped after \(maximumSteps) steps. Send another message to continue."))
      }
    }
  }

  // MARK: - Streaming

  private struct AgentTurn {
    let content: String
    let reasoning: String
    let toolCalls: [AgentToolCall]
    let finishReason: String?
  }

  private func streamTurn(onEvent: EventHandler) async throws -> AgentTurn {
    var messages: [[String: Any]] = [
      ["role": "system", "content": systemPrompt()]
    ]
    messages.append(contentsOf: history.map(\.jsonObject))

    var content = ""
    var reasoning = ""
    var toolCalls: [AgentToolCall] = []
    var finishReason: String?

    let tools = AgentToolCatalog.definitions(workspace: workspace.root)
    let stream: AsyncThrowingStream<AgentStreamEvent, Error>
    if let streamFactory {
      stream = streamFactory(modelID, messages, tools)
    } else if let client {
      stream = client.stream(model: modelID, messages: messages, tools: tools)
    } else {
      throw AgentRunnerError.notConfigured
    }

    for try await event in stream {
      if Task.isCancelled { throw CancellationError() }
      switch event {
      case .textDelta(let piece):
        content += piece
        onEvent(.textDelta(piece))
      case .reasoningDelta(let piece):
        reasoning += piece
        onEvent(.reasoningDelta(piece))
      case .completed(let finalContent, let finalCalls, let finish):
        if !finalContent.isEmpty { content = finalContent }
        toolCalls = finalCalls
        finishReason = finish
      }
    }

    return AgentTurn(
      content: content, reasoning: reasoning, toolCalls: toolCalls, finishReason: finishReason)
  }

  // MARK: - Context

  /// The configured context size converted to a conservative character budget.
  /// Reserves room for the system prompt, tool schemas, and the reply. Local
  /// models often run with a small context, so this is deliberately coarse.
  private var historyCharacterBudget: Int {
    let inputTokens = max(800, contextTokens * 7 / 10 - 1_800)
    return min(max(inputTokens * 3, 3_000), 160_000)
  }

  /// Keeps the first user turn, the current turn, and as much recent history as
  /// fits. Tool outputs are capped before whole messages are dropped.
  private func pruneHistory() {
    let budget = historyCharacterBudget
    let perMessageLimit = min(20_000, max(1_000, budget / 4))
    for index in history.indices where history[index].role == .tool {
      guard let content = history[index].content, content.count > perMessageLimit else {
        continue
      }
      history[index].content = String(content.prefix(perMessageLimit)) + "\n… [output truncated]"
    }

    let keepHead = history.first?.role == .user ? 1 : 0

    func removeOldest(excess: Bool) -> Bool {
      guard let lastUser = history.lastIndex(where: { $0.role == .user }) else { return false }
      let removeIndex = keepHead
      guard removeIndex < lastUser, removeIndex < history.count - (excess ? 1 : 2) else {
        return false
      }
      let removed = history.remove(at: removeIndex)
      if removed.role == .assistant, !removed.toolCalls.isEmpty {
        while removeIndex < history.count, history[removeIndex].role == .tool {
          history.remove(at: removeIndex)
        }
      }
      return true
    }

    while history.count > Self.maximumHistoryMessages {
      guard removeOldest(excess: false) else { break }
    }

    var total = history.reduce(0) { $0 + $1.estimatedCharacters }
    while total > budget {
      let before = history.count
      guard removeOldest(excess: true) else { break }
      total = history.reduce(0) { $0 + $1.estimatedCharacters }
      if history.count == before { break }
    }
  }

  private func systemPrompt() -> String {
    """
    You are Llama Agent, a coding agent running in the user's terminal on this Mac. \
    Help with software engineering tasks in their project using the tools provided.

    # Working directory
    All relative paths resolve against: \(workspace.displayPath)
    The file tools stay inside this folder. The shell starts there but can reach
    anything, so shell commands need the user's approval.

    # Tools
    - Read a file before editing it; use edit for targeted changes and write only for new files or full rewrites.
    - Prefer read, list, glob, and grep over shell commands when they can answer the question.
    - Use bash for builds, tests, git, and other commands without a dedicated tool.
    - After changing code, run the relevant check (build, test, lint) when one is available.
    - Writes and shell commands need the user's approval; if a call is denied, adjust instead of retrying it unchanged.

    # Communication
    - Responses render as Markdown. Keep them clear and concise.
    - Refer to files by path, and mention the outcome of commands you ran.
    - Keep changes consistent with the project's existing structure, naming, and style.
    - Treat unfamiliar files or changes as possible user work; investigate before overwriting or deleting them.
    - The agent works offline: don't suggest uploading code or using web services.
    """
  }
}
