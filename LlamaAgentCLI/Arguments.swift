import Darwin
import Foundation

enum CLICommand {
  case interactive
  case run
  case models
}

enum CLIFormat: String {
  case text
  case json
}

enum CLIError: LocalizedError {
  case usage(String)
  case server(String)
  case noModels
  case noInput

  var errorDescription: String? {
    switch self {
    case .usage(let message): return message
    case .server(let message): return message
    case .noModels: return "No models are available from the local server."
    case .noInput: return "No message provided. Pass one as an argument or pipe it on stdin."
    }
  }

  var exitCode: Int32 {
    switch self {
    case .usage: return 2
    default: return 1
    }
  }
}

struct CLIOptions {
  var command: CLICommand = .interactive
  var project: String?
  var message: [String] = []
  var model: String?
  var autoApprove = false
  var format: CLIFormat = .text
  var attach: String?
  var port: Int?
  var contextTokens: Int?
  var showHelp = false
  var showVersion = false
  var filter: String?

  static func parse(_ arguments: [String]) throws -> CLIOptions {
    var options = CLIOptions()
    var positionals: [String] = []
    var parsingMessage = false
    var index = 0

    func nextValue(_ flag: String) throws -> String {
      index += 1
      guard index < arguments.count else {
        throw CLIError.usage("Missing value for \(flag).")
      }
      return arguments[index]
    }

    while index < arguments.count {
      let argument = arguments[index]

      if argument == "--" {
        index += 1
        while index < arguments.count {
          if parsingMessage {
            options.message.append(arguments[index])
          } else {
            positionals.append(arguments[index])
          }
          index += 1
        }
        break
      }

      if argument == "-h" || argument == "--help" {
        options.showHelp = true
      } else if argument == "-v" || argument == "--version" {
        options.showVersion = true
      } else if argument == "--auto" || argument == "-a" {
        options.autoApprove = true
      } else if argument == "-m" || argument == "--model" {
        options.model = try nextValue(argument)
      } else if argument.hasPrefix("--model=") {
        options.model = String(argument.dropFirst("--model=".count))
      } else if argument == "--format" {
        options.format = try parseFormat(try nextValue(argument))
      } else if argument.hasPrefix("--format=") {
        options.format = try parseFormat(String(argument.dropFirst("--format=".count)))
      } else if argument == "--attach" {
        options.attach = try nextValue(argument)
      } else if argument.hasPrefix("--attach=") {
        options.attach = String(argument.dropFirst("--attach=".count))
      } else if argument == "--port" || argument == "-p" {
        options.port = try parsePort(try nextValue(argument))
      } else if argument.hasPrefix("--port=") {
        options.port = try parsePort(String(argument.dropFirst("--port=".count)))
      } else if argument == "--ctx" || argument == "--context" {
        options.contextTokens = try parseContext(try nextValue(argument))
      } else if argument.hasPrefix("--ctx=") {
        options.contextTokens = try parseContext(String(argument.dropFirst("--ctx=".count)))
      } else if argument.hasPrefix("--context=") {
        options.contextTokens = try parseContext(String(argument.dropFirst("--context=".count)))
      } else if parsingMessage {
        options.message.append(argument)
      } else if argument == "run" {
        options.command = .run
        parsingMessage = true
      } else if argument == "models" {
        options.command = .models
      } else if argument == "help" {
        options.showHelp = true
      } else if argument == "version" {
        options.showVersion = true
      } else if argument.hasPrefix("-") {
        throw CLIError.usage("Unknown option: \(argument)")
      } else {
        positionals.append(argument)
      }

      index += 1
    }

    if options.port != nil, options.attach != nil {
      throw CLIError.usage("Use either --attach or --port, not both.")
    }

    switch options.command {
    case .interactive:
      guard positionals.count <= 1 else {
        throw CLIError.usage("Unexpected argument: \(positionals[1]).")
      }
      options.project = positionals.first
    case .run:
      guard positionals.count <= 1 else {
        throw CLIError.usage("Put the project before `run`, or use --attach for the server.")
      }
      options.project = positionals.first
    case .models:
      guard positionals.count <= 1 else {
        throw CLIError.usage("Unexpected argument: \(positionals[1]).")
      }
      options.filter = positionals.first
    }

    if options.showHelp || options.showVersion {
      return options
    }
    if options.command == .run, options.message.isEmpty {
      options.message = []
    }
    return options
  }

  private static func parseFormat(_ value: String) throws -> CLIFormat {
    let normalized = value.lowercased()
    if normalized == "default" { return .text }
    guard let format = CLIFormat(rawValue: normalized) else {
      throw CLIError.usage("Unknown format: \(value). Use default or json.")
    }
    return format
  }

  private static func parsePort(_ value: String) throws -> Int {
    guard let port = Int(value), (1...65_535).contains(port) else {
      throw CLIError.usage("Invalid port: \(value).")
    }
    return port
  }

  private static func parseContext(_ value: String) throws -> Int {
    guard let tokens = Int(value), tokens >= 512 else {
      throw CLIError.usage("Invalid context size: \(value). Use at least 512.")
    }
    return tokens
  }
}

enum CLIHelp {
  static let text = """
    llama-agent — a local coding agent for the terminal

    Usage:
      llama-agent [project]                    Start the interactive agent
      llama-agent [project] run [message..]    Run one task and exit
      llama-agent models                       List models from the local server

    Options:
      -m, --model <id>       Model to use (defaults to the server's first model)
      -a, --auto             Allow writes and shell commands without asking
          --format <format>  Output format: default or json (run only)
          --attach <url>     llama-server base URL (default http://127.0.0.1:9931/v1)
      -p, --port <port>      llama-server port (default 9931)
          --ctx <tokens>     Context size used for history budgeting
      -h, --help             Show this help
      -v, --version          Show the version

    Environment:
      LLAMA_AGENT_URL        Base URL used when --attach/--port are omitted
      LLAMA_API_KEY          API key sent as a bearer token, when set

    Examples:
      llama-agent ~/code/app
      llama-agent ~/code/app run "Fix the failing tests"
      llama-agent run --auto --format json "Summarize the repository"
      llama-agent models
    """
}

enum CLIVersion {
  static let value = "0.1.0"
}
