import Foundation
import Observation

/// Drives one coding-agent conversation against llama-server. The loop is
/// deliberately small and open, in the spirit of OpenCode's build agent:
/// stream a completion, run any tools the model asks for, feed the results
/// back, and repeat until it answers. Nothing leaves the Mac.
@MainActor
@Observable
final class AgentSession {
  struct TranscriptEntry: Identifiable {
    enum Kind {
      case user, assistant, reasoning, toolCall, toolResult, notice, error
    }

    let id: UUID
    var kind: Kind
    var title: String?
    var text: String
    var detail: String?
    var isError: Bool
    var isPending: Bool

    init(
      id: UUID = UUID(), kind: Kind, title: String? = nil, text: String,
      detail: String? = nil, isError: Bool = false, isPending: Bool = false
    ) {
      self.id = id
      self.kind = kind
      self.title = title
      self.text = text
      self.detail = detail
      self.isError = isError
      self.isPending = isPending
    }
  }

  struct PendingApproval: Identifiable {
    let id = UUID()
    let toolName: String
    let title: String
    let detail: String
  }

  var entries: [TranscriptEntry] = []
  var isRunning = false
  var pendingApproval: PendingApproval?
  var scrollTick = 0
  var modelID: String
  var workspaceURL: URL?

  /// Test seam: replaces the HTTP client with a canned stream. Production
  /// sessions leave this nil and talk to llama-server.
  typealias StreamFactory =
    (String, [[String: Any]], [[String: Any]]) -> AsyncThrowingStream<AgentStreamEvent, Error>
  private let streamFactory: StreamFactory?

  private var history: [AgentMessage] = []
  private var runTask: Task<Void, Never>?
  private var approvalContinuation: CheckedContinuation<Bool, Never>?
  private var approvedTools: Set<String> = []

  private static let maximumSteps = 40
  private static let maximumHistoryMessages = 24
  private static let maximumHistoryCharacters = 120_000

  init(
    modelID: String? = nil, workspaceURL: URL? = nil, streamFactory: StreamFactory? = nil
  ) {
    let installed = ModelManager.shared.downloadedModels
    if let modelID, !modelID.isEmpty {
      self.modelID = modelID
    } else if let preferred = UserSettings.agentModelID,
      installed.contains(where: { $0.id == preferred })
    {
      self.modelID = preferred
    } else {
      self.modelID = installed.first?.id ?? ""
    }
    self.streamFactory = streamFactory

    if let workspaceURL,
      FileManager.default.fileExists(atPath: workspaceURL.path, isDirectory: nil)
    {
      self.workspaceURL = workspaceURL
    } else {
      self.workspaceURL = nil
    }
  }

  var workspace: AgentWorkspace? {
    workspaceURL.map { AgentWorkspace(root: $0) }
  }

  var canSend: Bool {
    !isRunning && workspace != nil && !modelID.isEmpty
  }

  // MARK: - Public actions

  func send(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, !isRunning else { return }
    guard let workspace else {
      append(.error, "Choose a project folder before sending a task.")
      return
    }
    guard !modelID.isEmpty else {
      append(.error, "Install a model first, then pick it in the toolbar.")
      return
    }

    history.append(AgentMessage(role: .user, content: trimmed))
    entries.append(TranscriptEntry(kind: .user, text: trimmed))
    pruneHistory()

    isRunning = true
    let model = modelID
    runTask = Task { [weak self, model, workspace] in
      await self?.run(model: model, workspace: workspace)
    }
  }

  func cancel() {
    runTask?.cancel()
    if let continuation = approvalContinuation {
      approvalContinuation = nil
      pendingApproval = nil
      continuation.resume(returning: false)
    }
  }

  func resolveApproval(allow: Bool, remember: Bool) {
    guard let continuation = approvalContinuation else { return }
    approvalContinuation = nil
    if allow, remember, let toolName = pendingApproval?.toolName {
      approvedTools.insert(toolName)
    }
    pendingApproval = nil
    continuation.resume(returning: allow)
  }

  func reset() {
    guard !isRunning else { return }
    entries.removeAll()
    history.removeAll()
    approvedTools.removeAll()
    pendingApproval = nil
  }

  func setWorkspace(_ url: URL?) {
    guard !isRunning else { return }
    workspaceURL = url
    UserSettings.agentProjectPath = url?.path
    reset()
  }

  func setModel(_ id: String) {
    guard !isRunning else { return }
    modelID = id
    UserSettings.agentModelID = id
  }

  // MARK: - Loop

  private func run(model: String, workspace: AgentWorkspace) async {
    defer {
      isRunning = false
      runTask = nil
      pendingApproval = nil
      approvalContinuation = nil
    }

    let baseURL = InferenceEngineRegistry.current.apiBaseURL
    if streamFactory == nil, baseURL == nil {
      append(.error, "The local server isn't ready yet. Wait for it to start and try again.")
      return
    }
    let client = baseURL.map {
      AgentClient(baseURL: $0, apiKey: LlamaServer.requestAPIKey)
    }

    for step in 1...Self.maximumSteps {
      if Task.isCancelled { return }

      let assistantID = UUID()
      entries.append(TranscriptEntry(id: assistantID, kind: .assistant, text: "", isPending: true))

      let turn: AgentTurn
      do {
        turn = try await streamTurn(
          model: model, workspace: workspace, client: client, assistantID: assistantID)
      } catch is CancellationError {
        entries.removeAll { $0.id == assistantID && $0.text.isEmpty }
        return
      } catch {
        entries.removeAll { $0.id == assistantID && $0.text.isEmpty }
        append(.error, error.localizedDescription)
        return
      }

      let content = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
      let reasoning = turn.reasoning.trimmingCharacters(in: .whitespacesAndNewlines)

      if content.isEmpty {
        entries.removeAll { $0.id == assistantID }
      } else {
        updateEntry(assistantID) {
          $0.text = content
          $0.isPending = false
        }
      }
      if !reasoning.isEmpty {
        let index = entries.firstIndex { $0.id == assistantID } ?? entries.count
        entries.insert(
          TranscriptEntry(kind: .reasoning, text: reasoning), at: min(index, entries.count))
        scrollTick &+= 1
      }

      if !content.isEmpty || !turn.toolCalls.isEmpty {
        history.append(
          AgentMessage(
            role: .assistant, content: content.isEmpty ? nil : content,
            toolCalls: turn.toolCalls))
      }

      guard !turn.toolCalls.isEmpty else {
        if turn.finishReason == "length" {
          append(.notice, "The model hit its output limit. Ask it to continue if the answer is incomplete.")
        }
        if content.isEmpty {
          append(.notice, "The model returned no text and made no tool calls.")
        }
        return
      }

      for call in turn.toolCalls {
        if Task.isCancelled { return }

        if AgentToolRunner.isMutating(call.name) {
          let allowed =
            approvedTools.contains(call.name) ? true : await approve(call, workspace: workspace)
          if !allowed {
            let denial = "The user denied this tool call."
            history.append(
              AgentMessage(
                role: .tool, content: denial, toolCallID: call.id, name: call.name))
            entries.append(
              TranscriptEntry(
                kind: .toolResult, title: call.name, text: denial, isError: true))
            scrollTick &+= 1
            continue
          }
        }

        let summary = AgentToolRunner.summary(for: call, workspace: workspace.root)
        entries.append(
          TranscriptEntry(kind: .toolCall, title: call.name, text: summary.title, detail: summary.detail)
        )
        scrollTick &+= 1

        let result = await AgentToolRunner.execute(call, workspace: workspace)
        history.append(
          AgentMessage(role: .tool, content: result.text, toolCallID: call.id, name: call.name))
        entries.append(
          TranscriptEntry(kind: .toolResult, title: call.name, text: result.text, isError: result.isError)
        )
        scrollTick &+= 1
      }

      pruneHistory()

      if step == Self.maximumSteps {
        append(.notice, "Stopped after \(Self.maximumSteps) steps. Send another message to continue.")
      }
    }
  }

  private struct AgentTurn {
    let content: String
    let reasoning: String
    let toolCalls: [AgentToolCall]
    let finishReason: String?
  }

  private func streamTurn(
    model: String, workspace: AgentWorkspace, client: AgentClient?, assistantID: UUID
  ) async throws -> AgentTurn {
    var messages: [[String: Any]] = [
      ["role": "system", "content": systemPrompt(workspace: workspace)]
    ]
    messages.append(contentsOf: history.map(\.jsonObject))

    var content = ""
    var reasoning = ""
    var toolCalls: [AgentToolCall] = []
    var finishReason: String?

    let tools = AgentToolCatalog.definitions(workspace: workspace.root)
    let stream: AsyncThrowingStream<AgentStreamEvent, Error>
    if let streamFactory {
      stream = streamFactory(model, messages, tools)
    } else if let client {
      stream = client.stream(model: model, messages: messages, tools: tools)
    } else {
      throw AgentClientError.notConfigured
    }
    for try await event in stream {
      if Task.isCancelled { throw CancellationError() }
      switch event {
      case .textDelta(let piece):
        content += piece
        updateEntry(assistantID) {
          $0.text = content
          $0.isPending = true
        }
      case .reasoningDelta(let piece):
        reasoning += piece
      case .completed(let finalContent, let finalCalls, let finish):
        if !finalContent.isEmpty { content = finalContent }
        toolCalls = finalCalls
        finishReason = finish
      }
    }
    return AgentTurn(
      content: content, reasoning: reasoning, toolCalls: toolCalls, finishReason: finishReason)
  }

  private func approve(_ call: AgentToolCall, workspace: AgentWorkspace) async -> Bool {
    let summary = AgentToolRunner.summary(for: call, workspace: workspace.root)
    pendingApproval = PendingApproval(
      toolName: call.name, title: summary.title, detail: summary.detail)
    let allowed = await withCheckedContinuation { continuation in
      approvalContinuation = continuation
    }
    pendingApproval = nil
    return allowed
  }

  // MARK: - Context

  /// The model's configured context size, converted to a conservative character
  /// budget. Reserves room for the system prompt, the tool schemas, and the
  /// model's reply; local models default to 4k, so this can be much smaller
  /// than the fixed cap used for large-context models.
  private var historyCharacterBudget: Int {
    let contextTokens =
      ModelManager.shared.downloadedModels
      .first { $0.id == modelID }?.effectiveCtxTier?.rawValue ?? 8_192
    let inputTokens = max(800, contextTokens * 7 / 10 - 1_800)
    return min(max(inputTokens * 3, 3_000), 160_000)
  }

  /// Keeps the first user turn (the task), the current turn, and as much recent
  /// history as fits. Tool outputs are the biggest contributors, so they're
  /// capped before whole messages are dropped. This is a coarse guard, not a
  /// tokenizer.
  private func pruneHistory() {
    let budget = historyCharacterBudget
    let perMessageLimit = min(20_000, max(1_000, budget / 4))
    for index in history.indices where history[index].role == .tool {
      guard let content = history[index].content, content.count > perMessageLimit else {
        continue
      }
      history[index].content =
        String(content.prefix(perMessageLimit)) + "\n… [output truncated]"
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
      let after = history.reduce(0) { $0 + $1.estimatedCharacters }
      total = after
      if history.count == before { break }
    }
  }

  private func systemPrompt(workspace: AgentWorkspace) -> String {
    """
    You are Llama Agent, a coding agent running inside the Llama app on this Mac. \
    Help the user with software engineering tasks in their project using the tools provided.

    # Working directory
    All relative paths resolve against: \(workspace.displayPath)
    The file tools stay inside this folder. The shell starts there but can reach
    anything, so every shell command needs the user's approval.

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
    - This agent works offline: don't suggest uploading code or using web services.
    """
  }

  // MARK: - Transcript helpers

  private func updateEntry(_ id: UUID, _ mutate: (inout TranscriptEntry) -> Void) {
    guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
    mutate(&entries[index])
    scrollTick &+= 1
  }

  private func append(_ kind: TranscriptEntry.Kind, _ text: String) {
    entries.append(
      TranscriptEntry(kind: kind, text: text, isError: kind == .error))
    scrollTick &+= 1
  }
}
