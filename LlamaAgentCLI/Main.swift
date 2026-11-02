import Darwin
import Foundation

@main
struct LlamaAgentCLI {
  static func main() async {
    let options: CLIOptions
    do {
      options = try CLIOptions.parse(Array(CommandLine.arguments.dropFirst()))
    } catch {
      let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
      Terminal.error("error: \(message)")
      exit((error as? CLIError)?.exitCode ?? 2)
    }

    if options.showHelp {
      print(CLIHelp.text)
      return
    }
    if options.showVersion {
      print("llama-agent \(CLIVersion.value)")
      return
    }

    do {
      try await CLIApp(options: options).run()
    } catch let error as CLIError {
      report(error.localizedDescription, format: options.format)
      exit(error.exitCode)
    } catch is CancellationError {
      exit(130)
    } catch {
      report(error.localizedDescription, format: options.format)
      exit(1)
    }
  }

  private static func report(_ message: String, format: CLIFormat) {
    if format == .json {
      print(jsonString(["error": message]))
    } else {
      Terminal.error("error: \(message)")
    }
  }
}

private func jsonString(_ object: Any) -> String {
  guard
    let data = try? JSONSerialization.data(
      withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
    let text = String(data: data, encoding: .utf8)
  else {
    return "{}"
  }
  return text
}

struct CLIApp {
  let options: CLIOptions
  private let environment = ProcessInfo.processInfo.environment

  func run() async throws {
    let baseURL = try CLIConfiguration.baseURL(options: options, environment: environment)
    let apiKey = CLIConfiguration.apiKey(environment: environment)
    let server = ServerClient(baseURL: baseURL, apiKey: apiKey)

    switch options.command {
    case .models:
      try await listModels(server)
    case .run:
      try await runOnce(server: server, baseURL: baseURL, apiKey: apiKey)
    case .interactive:
      let workspace = try CLIConfiguration.workspace(options.project)
      let model = try await CLIConfiguration.model(
        options: options, server: server, environment: environment)
      let runner = AgentRunner(
        modelID: model,
        workspaceURL: workspace,
        baseURL: baseURL,
        apiKey: apiKey,
        contextTokens: CLIConfiguration.contextTokens(options: options, environment: environment))
      let session = InteractiveSession(
        runner: runner, server: server, baseURL: baseURL, apiKey: apiKey)
      try await session.run()
    }
  }

  private func listModels(_ server: ServerClient) async throws {
    var models = try await server.models()
    if let filter = options.filter, !filter.isEmpty {
      models = models.filter { $0.localizedCaseInsensitiveContains(filter) }
    }

    if options.format == .json {
      print(jsonString(["models": models]))
    } else {
      for model in models {
        print(model)
      }
    }
  }

  private func runOnce(server: ServerClient, baseURL: URL, apiKey: String?) async throws {
    let workspace = try CLIConfiguration.workspace(options.project)
    let model = try await CLIConfiguration.model(
      options: options, server: server, environment: environment)

    var message = options.message.joined(separator: " ")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    if message.isEmpty {
      guard !Terminal.stdinIsTTY else { throw CLIError.noInput }
      let data = FileHandle.standardInput.readDataToEndOfFile()
      message = String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
    guard !message.isEmpty else { throw CLIError.noInput }

    let runner = AgentRunner(
      modelID: model,
      workspaceURL: workspace,
      baseURL: baseURL,
      apiKey: apiKey,
      contextTokens: CLIConfiguration.contextTokens(options: options, environment: environment),
      autoApprove: options.autoApprove)

    let sink = ConsoleSink(format: options.format)
    let approval: AgentRunner.ApprovalHandler? =
      options.autoApprove ? nil : { request in InteractiveSession.promptApproval(request) }

    do {
      try await runner.send(
        message, onEvent: { sink.handle($0) }, approval: approval)
    } catch {
      sink.finish()
      throw error
    }
    sink.finish()

    if options.format == .json {
      print(sink.jsonOutput(model: model, workspace: workspace.path))
    }
  }
}

private final class ConsoleSink {
  let format: CLIFormat
  private var renderer = OutputRenderer()
  private var streamedText = ""
  private var completedText = ""
  private var sawTextThisTurn = false
  private var tools: [[String: Any]] = []
  private var notices: [String] = []
  private var reasoning = ""

  init(format: CLIFormat) {
    self.format = format
  }

  func handle(_ event: AgentEvent) {
    switch event {
    case .assistantStarted:
      sawTextThisTurn = false
    case .textDelta(let piece):
      streamedText += piece
      sawTextThisTurn = true
      if format == .text {
        renderer.text(piece)
      }
    case .reasoningDelta(let piece):
      reasoning += piece
    case .assistantCompleted(let content):
      if !content.isEmpty { completedText = content }
      if format == .text, !sawTextThisTurn, !content.isEmpty {
        renderer.text(content)
      }
    case .toolCall(let name, let title, let detail):
      if format == .text {
        renderer.toolCall(title: title)
      }
      tools.append(["name": name, "title": title, "detail": detail])
    case .toolResult(let name, let text, let isError):
      if format == .text {
        renderer.toolResult(text, isError: isError)
      }
      if let index = tools.lastIndex(where: {
        $0["result"] == nil && ($0["name"] as? String) == name
      }) {
        tools[index]["result"] = Terminal.truncate(text, limit: 20_000)
        tools[index]["isError"] = isError
      } else {
        tools.append([
          "name": name, "title": name, "detail": "",
          "result": Terminal.truncate(text, limit: 20_000), "isError": isError,
        ])
      }
    case .notice(let text):
      notices.append(text)
      if format == .text {
        renderer.status(Terminal.yellow("notice: \(text)"))
      }
    }
  }

  func finish() {
    if format == .text {
      renderer.finishLine()
    }
  }

  func jsonOutput(model: String, workspace: String) -> String {
    var payload: [String: Any] = [
      "model": model,
      "workspace": workspace,
      "text": completedText.isEmpty ? streamedText : completedText,
    ]
    if !reasoning.isEmpty { payload["reasoning"] = reasoning }
    if !tools.isEmpty { payload["tools"] = tools }
    if !notices.isEmpty { payload["notices"] = notices }
    return jsonString(payload)
  }
}

private final class InteractiveSession {
  private var runner: AgentRunner
  private let server: ServerClient
  private let baseURL: URL
  private let apiKey: String?
  private var isRunning = true

  init(runner: AgentRunner, server: ServerClient, baseURL: URL, apiKey: String?) {
    self.runner = runner
    self.server = server
    self.baseURL = baseURL
    self.apiKey = apiKey
  }

  func run() async throws {
    print("\(Terminal.bold("llama-agent")) \(CLIVersion.value)")
    print("model:     \(runner.modelID)")
    print("workspace: \(runner.workspace.displayPath)")
    print("Type /help for commands, /exit to quit.\n")

    while isRunning {
      guard let line = Terminal.readLine(prompt: Terminal.bold("llama-agent> ")) else {
        break
      }
      let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmed.isEmpty else { continue }

      if trimmed.hasPrefix("/") {
        await handleCommand(trimmed)
      } else {
        await send(trimmed)
      }
    }
  }

  private func send(_ text: String) async {
    let sink = ConsoleSink(format: .text)
    Terminal.write("\n", to: .standardError)
    do {
      try await runner.send(
        text,
        onEvent: { sink.handle($0) },
        approval: { request in InteractiveSession.promptApproval(request) })
    } catch is CancellationError {
      sink.finish()
      Terminal.error("cancelled")
    } catch {
      sink.finish()
      Terminal.error("error: \(error.localizedDescription)")
    }
    sink.finish()
  }

  private func handleCommand(_ command: String) async {
    let parts = command.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
    let name = parts.first.map(String.init)?.lowercased() ?? ""
    let argument = parts.count > 1
      ? String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines) : ""

    switch name {
    case "/exit", "/quit", "/q":
      isRunning = false
    case "/help":
      print(Self.helpText)
    case "/clear":
      runner.reset()
      print("History cleared.")
    case "/model":
      if argument.isEmpty {
        print("model: \(runner.modelID)")
      } else {
        runner.modelID = argument
        runner.reset()
        print("model: \(runner.modelID) (history cleared)")
      }
    case "/workspace":
      if argument.isEmpty {
        print("workspace: \(runner.workspace.displayPath)")
      } else {
        do {
          let workspace = try CLIConfiguration.workspace(argument)
          runner = makeRunner(model: runner.modelID, workspace: workspace)
          print("workspace: \(runner.workspace.displayPath) (history cleared)")
        } catch {
          Terminal.error("error: \(error.localizedDescription)")
        }
      }
    case "/auto":
      switch argument.lowercased() {
      case "", "on", "true", "yes", "1":
        runner.autoApprove = true
        print("auto-approve: on")
      case "off", "false", "no", "0":
        runner.autoApprove = false
        print("auto-approve: off")
      default:
        Terminal.error("usage: /auto [on|off]")
      }
    case "/ctx":
      if argument.isEmpty {
        print("context: \(runner.contextTokens) tokens")
      } else if let tokens = Int(argument), tokens >= 512 {
        runner.contextTokens = tokens
        print("context: \(runner.contextTokens) tokens")
      } else {
        Terminal.error("usage: /ctx <tokens> (minimum 512)")
      }
    case "/status":
      print("model:      \(runner.modelID)")
      print("workspace:  \(runner.workspace.displayPath)")
      print("context:    \(runner.contextTokens) tokens")
      print("auto:       \(runner.autoApprove ? "on" : "off")")
    case "/models":
      do {
        let models = try await server.models()
        print(models.joined(separator: "\n"))
      } catch {
        Terminal.error("error: \(error.localizedDescription)")
      }
    default:
      Terminal.error("Unknown command: \(name). Type /help for the list.")
    }
  }

  private func makeRunner(model: String, workspace: URL) -> AgentRunner {
    AgentRunner(
      modelID: model,
      workspaceURL: workspace,
      baseURL: baseURL,
      apiKey: apiKey,
      contextTokens: runner.contextTokens,
      autoApprove: runner.autoApprove)
  }

  static func promptApproval(_ request: AgentApprovalRequest) -> AgentApprovalDecision {
    guard Terminal.stdinIsTTY else { return .deny }
    Terminal.write("\n", to: .standardError)
    Terminal.write("\(request.title)\n", to: .standardError)
    if !request.detail.isEmpty {
      Terminal.write(Terminal.dim(request.detail) + "\n", to: .standardError)
    }
    guard let answer = Terminal.readLine(prompt: "Approve? [y/N/a] ") else { return .deny }
    switch answer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "y", "yes": return .allow
    case "a", "always": return .allowAlways
    default: return .deny
    }
  }

  private static let helpText = """
    Commands:
      /help             Show this help
      /model [id]       Show or switch the model
      /models           List models from the server
      /workspace [dir]  Show or switch the project folder
      /auto [on|off]    Toggle write/shell approvals
      /ctx [tokens]     Show or set the context budget
      /clear            Clear the conversation history
      /status           Show the current settings
      /exit             Quit
    """
}
