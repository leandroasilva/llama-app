import Foundation

/// A message in the agent's OpenAI-compatible chat history. Kept separate from
/// the transcript the UI renders: the model only needs roles, text, and tool
/// calls, while the UI also shows reasoning and tool results as their own rows.
struct AgentMessage: Identifiable, Equatable {
  enum Role: String {
    case system, user, assistant, tool
  }

  let id: UUID
  var role: Role
  var content: String?
  var toolCalls: [AgentToolCall]
  var toolCallID: String?
  var name: String?

  init(
    id: UUID = UUID(),
    role: Role,
    content: String? = nil,
    toolCalls: [AgentToolCall] = [],
    toolCallID: String? = nil,
    name: String? = nil
  ) {
    self.id = id
    self.role = role
    self.content = content
    self.toolCalls = toolCalls
    self.toolCallID = toolCallID
    self.name = name
  }

  /// The wire shape llama-server expects. Assistant turns with tool calls keep
  /// `content` as an empty string: null is accepted by OpenAI but some local
  /// chat templates treat it differently from empty.
  var jsonObject: [String: Any] {
    switch role {
    case .assistant:
      var object: [String: Any] = ["role": "assistant", "content": content ?? ""]
      if !toolCalls.isEmpty {
        object["tool_calls"] = toolCalls.map(\.jsonObject)
      }
      return object
    case .tool:
      return [
        "role": "tool",
        "tool_call_id": toolCallID ?? "",
        "content": content ?? "",
      ]
    case .system, .user:
      var object: [String: Any] = ["role": role.rawValue, "content": content ?? ""]
      if let name { object["name"] = name }
      return object
    }
  }

  /// Rough size used for context budgeting. Token counts aren't available
  /// without a tokenizer, and characters are close enough for a coarse cap.
  var estimatedCharacters: Int {
    (content?.count ?? 0) + toolCalls.reduce(0) { $0 + $1.name.count + $1.arguments.count }
  }
}

/// One function call the model asked the harness to run.
struct AgentToolCall: Identifiable, Equatable {
  var id: String
  var name: String
  var arguments: String

  var jsonObject: [String: Any] {
    [
      "id": id,
      "type": "function",
      "function": ["name": name, "arguments": arguments],
    ]
  }

  var argumentObject: [String: Any] {
    guard let data = arguments.data(using: .utf8),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [:] }
    return object
  }
}

/// Result of running one tool, already formatted for the model.
struct AgentToolOutput {
  let text: String
  let isError: Bool

  static func failure(_ message: String) -> AgentToolOutput {
    AgentToolOutput(text: message, isError: true)
  }
}

enum AgentToolError: LocalizedError {
  case invalidArguments(String)
  case outsideWorkspace(String)
  case notFound(String)
  case failed(String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments(let message): return message
    case .outsideWorkspace(let path): return "Path is outside the project folder: \(path)"
    case .notFound(let path): return "No such file or directory: \(path)"
    case .failed(let message): return message
    }
  }
}

/// The tool schemas advertised to the model. Descriptions are deliberately
/// short: small local models follow a handful of explicit rules better than a
/// long manual, and every token here is re-sent on each step.
enum AgentToolCatalog {
  static func definitions(workspace: URL?) -> [[String: Any]] {
    let root = workspace?.path ?? "the project folder"
    return [
      tool(
        name: "read",
        description:
          "Read a text file from the project. Returns numbered lines. Use offset and limit for large files.",
        properties: [
          "path": string("Path relative to \(root)."),
          "offset": integer("First line to read, 1-based. Defaults to 1."),
          "limit": integer("Maximum number of lines. Defaults to 2000."),
        ],
        required: ["path"]),
      tool(
        name: "write",
        description:
          "Create a file or replace its entire contents. Prefer edit for targeted changes.",
        properties: [
          "path": string("Path relative to \(root)."),
          "content": string("Full file contents."),
        ],
        required: ["path", "content"]),
      tool(
        name: "edit",
        description:
          "Replace an exact string in a text file. oldString must occur exactly once unless replaceAll is true, and must differ from newString.",
        properties: [
          "path": string("Path relative to \(root)."),
          "oldString": string("Exact text to replace."),
          "newString": string("Replacement text."),
          "replaceAll": boolean("Replace every occurrence instead of requiring a unique match."),
        ],
        required: ["path", "oldString", "newString"]),
      tool(
        name: "bash",
        description:
          "Run a shell command in the project folder. Returns combined stdout/stderr and the exit code. Use for builds, tests, git, and other commands that have no dedicated tool.",
        properties: [
          "command": string("Command to run with /bin/zsh -lc."),
          "description": string("Short description of what the command does."),
          "timeout": integer("Seconds before the command is stopped. Defaults to 120, maximum 600."),
        ],
        required: ["command"]),
      tool(
        name: "glob",
        description:
          "Find files by glob pattern. Supports *, **, and ?. Results are paths relative to the project.",
        properties: [
          "pattern": string("Glob pattern, for example **/*.swift."),
          "path": string("Directory to search, relative to the project. Defaults to the project root."),
        ],
        required: ["pattern"]),
      tool(
        name: "grep",
        description:
          "Search file contents with a regular expression. Returns matching paths with line numbers.",
        properties: [
          "pattern": string("Regular expression to search for."),
          "path": string("Directory or file to search, relative to the project. Defaults to the project root."),
          "include": string("Optional glob limiting which files are searched, for example *.swift."),
        ],
        required: ["pattern"]),
      tool(
        name: "list",
        description: "List the entries in a directory of the project.",
        properties: [
          "path": string("Directory relative to the project. Defaults to the project root.")
        ],
        required: []),
    ]
  }

  private static func tool(
    name: String, description: String, properties: [String: Any], required: [String]
  ) -> [String: Any] {
    [
      "type": "function",
      "function": [
        "name": name,
        "description": description,
        "parameters": [
          "type": "object",
          "properties": properties,
          "required": required,
          "additionalProperties": false,
        ],
      ],
    ]
  }

  private static func string(_ description: String) -> [String: Any] {
    ["type": "string", "description": description]
  }

  private static func integer(_ description: String) -> [String: Any] {
    ["type": "integer", "description": description]
  }

  private static func boolean(_ description: String) -> [String: Any] {
    ["type": "boolean", "description": description]
  }
}
