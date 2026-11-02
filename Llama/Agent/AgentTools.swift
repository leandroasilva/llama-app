import Darwin
import Foundation

/// Confines the agent's file tools to one project folder. Relative paths are
/// resolved against the root; absolute paths are accepted only when they land
/// inside it. Symlinks are resolved before the check so a link can't point the
/// agent outside the project.
struct AgentWorkspace {
  let root: URL

  init(root: URL) {
    self.root = root.standardizedFileURL.resolvingSymlinksInPath()
  }

  var displayPath: String { root.path }

  func resolve(_ path: String) throws -> URL {
    let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      throw AgentToolError.invalidArguments("A path is required.")
    }

    let expanded: String
    if trimmed == "~" {
      expanded = NSHomeDirectory()
    } else if trimmed.hasPrefix("~/") {
      expanded = NSHomeDirectory() + String(trimmed.dropFirst())
    } else {
      expanded = trimmed
    }

    let candidate = URL(
      fileURLWithPath: expanded, relativeTo: root
    ).standardizedFileURL.resolvingSymlinksInPath()

    let rootPath = root.path
    guard candidate.path == rootPath || candidate.path.hasPrefix(rootPath + "/") else {
      throw AgentToolError.outsideWorkspace(trimmed)
    }
    return candidate
  }

  func relativePath(for url: URL) -> String {
    let path = url.standardizedFileURL.path
    let rootPath = root.path
    guard path.hasPrefix(rootPath + "/") else { return url.lastPathComponent }
    return String(path.dropFirst(rootPath.count + 1))
  }
}

/// Executes the tools advertised in `AgentToolCatalog`. Reads run without
/// approval; writes, edits, and shell commands are gated by the session.
enum AgentToolRunner {
  static func isMutating(_ name: String) -> Bool {
    name == "write" || name == "edit" || name == "bash"
  }

  static func summary(for call: AgentToolCall, workspace: URL?) -> (title: String, detail: String) {
    let arguments = call.argumentObject
    let path = arguments["path"] as? String ?? ""
    switch call.name {
    case "write":
      let content = arguments["content"] as? String ?? ""
      return ("Write \(path)", preview(content, limit: 3000))
    case "edit":
      let old = arguments["oldString"] as? String ?? ""
      let new = arguments["newString"] as? String ?? ""
      return (
        "Edit \(path)",
        "Replace:\n\(preview(old, limit: 1200))\n\nWith:\n\(preview(new, limit: 1200))"
      )
    case "bash":
      return (
        "Run command",
        "\(arguments["command"] as? String ?? "")\n\nWorking directory: \(workspace?.path ?? "none")"
      )
    default:
      return (call.name, path)
    }
  }

  static func execute(_ call: AgentToolCall, workspace: AgentWorkspace, timeout: TimeInterval = 120)
    async -> AgentToolOutput
  {
    let arguments = call.argumentObject
    do {
      switch call.name {
      case "read":
        return try read(arguments, workspace: workspace)
      case "write":
        return try write(arguments, workspace: workspace)
      case "edit":
        return try edit(arguments, workspace: workspace)
      case "bash":
        return await bash(arguments, workspace: workspace, defaultTimeout: timeout)
      case "glob":
        return try glob(arguments, workspace: workspace)
      case "grep":
        return try grep(arguments, workspace: workspace)
      case "list":
        return try list(arguments, workspace: workspace)
      default:
        return .failure("Unknown tool: \(call.name)")
      }
    } catch let error as AgentToolError {
      return .failure(error.errorDescription ?? "Tool failed.")
    } catch {
      return .failure(error.localizedDescription)
    }
  }

  // MARK: - File tools

  private static let maximumFileBytes: Int64 = 8 * 1024 * 1024
  private static let skippedDirectories: Set<String> = [
    ".git", ".svn", ".hg", "node_modules", "Pods", "DerivedData", ".build",
    "build", "dist", "target", ".next", ".venv", "venv", "__pycache__",
  ]

  private static func read(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let url = try workspace.resolve(try requiredString("path", in: arguments))
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw AgentToolError.notFound(workspace.relativePath(for: url))
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
    guard size <= maximumFileBytes else {
      throw AgentToolError.failed(
        "File is too large to read (\(size) bytes). Read a smaller file or use grep.")
    }
    guard let contents = try? String(contentsOf: url, encoding: .utf8) else {
      throw AgentToolError.failed("File isn't valid UTF-8 text: \(workspace.relativePath(for: url))")
    }

    guard !contents.isEmpty else {
      return AgentToolOutput(text: "(empty file)", isError: false)
    }

    let lines = contents.components(separatedBy: "\n")
    let offset = max(1, optionalInt("offset", in: arguments) ?? 1)
    let limit = min(max(1, optionalInt("limit", in: arguments) ?? 2000), 4000)
    guard offset <= lines.count else {
      return AgentToolOutput(
        text: "Offset \(offset) is past the end of the file (\(lines.count) lines).", isError: true)
    }

    let end = min(lines.count, offset - 1 + limit)
    var output = ""
    var truncated = false
    for index in (offset - 1)..<end {
      let line = truncateLine(lines[index], limit: 1000)
      let numbered = String(format: "%6d  ", index + 1) + line
      if output.count + numbered.count > 120_000 {
        truncated = true
        break
      }
      output += numbered + "\n"
    }
    if end < lines.count {
      output += "\n[\(lines.count - end) more lines]"
    }
    if truncated {
      output += "\n[output truncated]"
    }
    return AgentToolOutput(text: output.isEmpty ? "(empty file)" : output, isError: false)
  }

  private static func write(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let relative = try requiredString("path", in: arguments)
    let content = try requiredString("content", in: arguments)
    guard content.utf8.count <= 2 * 1024 * 1024 else {
      throw AgentToolError.failed("Refusing to write more than 2 MB in one call.")
    }

    let url = try workspace.resolve(relative)
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    {
      throw AgentToolError.failed("Path is a directory: \(relative)")
    }

    let existed = FileManager.default.fileExists(atPath: url.path)
    let permissions = existingPermissions(at: url)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try content.write(to: url, atomically: true, encoding: .utf8)
    restore(permissions, to: url)
    return AgentToolOutput(
      text: "\(existed ? "Updated" : "Created") \(workspace.relativePath(for: url)) (\(content.utf8.count) bytes).",
      isError: false)
  }

  private static func edit(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let relative = try requiredString("path", in: arguments)
    let oldString = try requiredString("oldString", in: arguments)
    let newString = try requiredString("newString", in: arguments)
    guard oldString != newString else {
      throw AgentToolError.invalidArguments("oldString and newString must differ.")
    }
    guard !oldString.isEmpty else {
      throw AgentToolError.invalidArguments("oldString can't be empty.")
    }

    let url = try workspace.resolve(relative)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw AgentToolError.notFound(relative)
    }
    guard let contents = try? String(contentsOf: url, encoding: .utf8) else {
      throw AgentToolError.failed("File isn't valid UTF-8 text: \(relative)")
    }

    let occurrences = countOccurrences(of: oldString, in: contents)
    guard occurrences > 0 else {
      throw AgentToolError.failed(
        "oldString wasn't found in \(relative). Read the file and match the text exactly.")
    }
    let replaceAll = optionalBool("replaceAll", in: arguments) ?? false
    guard occurrences == 1 || replaceAll else {
      throw AgentToolError.failed(
        "oldString occurs \(occurrences) times in \(relative). Add surrounding context or set replaceAll.")
    }

    let updated: String
    if replaceAll {
      updated = contents.replacingOccurrences(of: oldString, with: newString)
    } else {
      guard let range = contents.range(of: oldString) else {
        throw AgentToolError.failed("Couldn't locate oldString in \(relative).")
      }
      updated = contents.replacingCharacters(in: range, with: newString)
    }
    let permissions = existingPermissions(at: url)
    try updated.write(to: url, atomically: true, encoding: .utf8)
    restore(permissions, to: url)
    return AgentToolOutput(
      text: "Edited \(relative) (\(occurrences) replacement\(occurrences == 1 ? "" : "s")).",
      isError: false)
  }

  private static func list(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let relative = optionalString("path", in: arguments) ?? "."
    let url = try workspace.resolve(relative)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
      throw AgentToolError.notFound(relative)
    }
    guard isDirectory.boolValue else {
      return AgentToolOutput(text: workspace.relativePath(for: url), isError: false)
    }

    let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]
    let contents = try FileManager.default.contentsOfDirectory(
      at: url, includingPropertiesForKeys: keys, options: [])
    let sorted = contents.sorted { lhs, rhs in
      let lhsDirectory = (try? lhs.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
      let rhsDirectory = (try? rhs.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
      if lhsDirectory != rhsDirectory { return lhsDirectory }
      return lhs.lastPathComponent.localizedStandardCompare(rhs.lastPathComponent) == .orderedAscending
    }

    var lines: [String] = []
    for item in sorted.prefix(500) {
      let values = try? item.resourceValues(forKeys: Set(keys))
      if values?.isDirectory == true {
        lines.append(item.lastPathComponent + "/")
      } else {
        let size = values?.fileSize ?? 0
        lines.append("\(item.lastPathComponent) (\(size) bytes)")
      }
    }
    if sorted.count > 500 { lines.append("… \(sorted.count - 500) more entries") }
    return AgentToolOutput(
      text: lines.isEmpty ? "(empty directory)" : lines.joined(separator: "\n"), isError: false)
  }

  // MARK: - Search tools

  private static func glob(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let pattern = try requiredString("pattern", in: arguments)
    let relative = optionalString("path", in: arguments) ?? "."
    let base = try workspace.resolve(relative)
    let regex = try globRegex(pattern)
    let matchesPattern = { (path: String) -> Bool in
      let range = NSRange(path.startIndex..<path.endIndex, in: path)
      if regex.firstMatch(in: path, options: [], range: range) != nil { return true }
      if !pattern.contains("/") {
        let name = (path as NSString).lastPathComponent
        let nameRange = NSRange(name.startIndex..<name.endIndex, in: name)
        return regex.firstMatch(in: name, options: [], range: nameRange) != nil
      }
      return false
    }

    var results: [String] = []
    let enumerator = FileManager.default.enumerator(
      at: base, includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles, .skipsPackageDescendants])
    while let item = enumerator?.nextObject() as? URL {
      let values = try? item.resourceValues(forKeys: [.isDirectoryKey])
      if values?.isDirectory == true {
        if skippedDirectories.contains(item.lastPathComponent) {
          enumerator?.skipDescendants()
        }
        continue
      }
      let path = workspace.relativePath(for: item)
      if matchesPattern(path) {
        results.append(path)
        if results.count >= 300 { break }
      }
    }

    results.sort { $0.localizedStandardCompare($1) == .orderedAscending }
    guard !results.isEmpty else {
      return AgentToolOutput(text: "No files matched \(pattern).", isError: false)
    }
    var text = results.joined(separator: "\n")
    if results.count >= 300 { text += "\n[results truncated at 300]" }
    return AgentToolOutput(text: text, isError: false)
  }

  private static func grep(_ arguments: [String: Any], workspace: AgentWorkspace) throws
    -> AgentToolOutput
  {
    let pattern = try requiredString("pattern", in: arguments)
    let relative = optionalString("path", in: arguments) ?? "."
    let base = try workspace.resolve(relative)
    let expression: NSRegularExpression
    do {
      expression = try NSRegularExpression(pattern: pattern)
    } catch {
      throw AgentToolError.invalidArguments("Invalid regular expression: \(error.localizedDescription)")
    }
    let include = optionalString("include", in: arguments)
    let includeRegex = try include.map(globRegex)

    var files: [URL] = []
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: base.path, isDirectory: &isDirectory) else {
      throw AgentToolError.notFound(relative)
    }
    if isDirectory.boolValue {
      let enumerator = FileManager.default.enumerator(
        at: base, includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants])
      while let item = enumerator?.nextObject() as? URL {
        let values = try? item.resourceValues(forKeys: [.isDirectoryKey])
        if values?.isDirectory == true {
          if skippedDirectories.contains(item.lastPathComponent) {
            enumerator?.skipDescendants()
          }
          continue
        }
        files.append(item)
      }
    } else {
      files = [base]
    }

    var results: [String] = []
    for file in files {
      guard results.count < 200 else { break }
      let path = workspace.relativePath(for: file)
      if let includeRegex {
        let range = NSRange(path.startIndex..<path.endIndex, in: path)
        let name = (path as NSString).lastPathComponent
        let nameRange = NSRange(name.startIndex..<name.endIndex, in: name)
        let matchesInclude =
          includeRegex.firstMatch(in: path, options: [], range: range) != nil
          || (!(include ?? "").contains("/")
            && includeRegex.firstMatch(in: name, options: [], range: nameRange) != nil)
        guard matchesInclude else { continue }
      }
      guard let data = try? Data(contentsOf: file, options: .mappedIfSafe),
        data.count <= 2 * 1024 * 1024,
        !data.prefix(8192).contains(0),
        let contents = String(data: data, encoding: .utf8)
      else { continue }

      let lines = contents.components(separatedBy: "\n")
      for (index, line) in lines.enumerated() {
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        if expression.firstMatch(in: line, options: [], range: range) != nil {
          results.append("\(path):\(index + 1): \(truncateLine(line, limit: 400))")
          if results.count >= 200 { break }
        }
      }
    }

    guard !results.isEmpty else {
      return AgentToolOutput(text: "No matches for \(pattern).", isError: false)
    }
    var text = results.joined(separator: "\n")
    if results.count >= 200 { text += "\n[results truncated at 200]" }
    return AgentToolOutput(text: text, isError: false)
  }

  // MARK: - Shell

  private static func bash(
    _ arguments: [String: Any], workspace: AgentWorkspace, defaultTimeout: TimeInterval
  ) async -> AgentToolOutput {
    guard let command = optionalString("command", in: arguments), !command.isEmpty else {
      return .failure("A command is required.")
    }
    let timeout = min(max(TimeInterval(optionalInt("timeout", in: arguments) ?? Int(defaultTimeout)), 1), 600)
    return await runShell(command: command, directory: workspace.root, timeout: timeout)
  }

  private static func runShell(command: String, directory: URL, timeout: TimeInterval) async
    -> AgentToolOutput
  {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/zsh")
    process.arguments = ["-lc", command]
    process.currentDirectoryURL = directory
    var environment = ProcessInfo.processInfo.environment
    environment["TERM"] = "dumb"
    environment["PAGER"] = "cat"
    environment["GIT_PAGER"] = "cat"
    environment["NO_COLOR"] = "1"
    process.environment = environment

    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    process.standardInput = FileHandle.nullDevice

    let state = ShellState()
    let outputHandle = stdout.fileHandleForReading
    let errorHandle = stderr.fileHandleForReading
    outputHandle.readabilityHandler = { handle in
      let data = handle.availableData
      if !data.isEmpty { state.append(data, isError: false) }
    }
    errorHandle.readabilityHandler = { handle in
      let data = handle.availableData
      if !data.isEmpty { state.append(data, isError: true) }
    }

    return await withTaskCancellationHandler {
      await withCheckedContinuation { continuation in
        state.continuation = continuation
        process.terminationHandler = { process in
          let status = process.terminationStatus
          // Let the readability handlers drain the final bytes. Reading the
          // pipes to EOF here can block forever when the command left a child
          // process holding them open, so stop after a short grace period.
          DispatchQueue.global().asyncAfter(deadline: .now() + 0.15) {
            outputHandle.readabilityHandler = nil
            errorHandle.readabilityHandler = nil
            state.finish(status: status)
          }
        }
        do {
          try process.run()
        } catch {
          outputHandle.readabilityHandler = nil
          errorHandle.readabilityHandler = nil
          state.finish(error: "Couldn't start the command: \(error.localizedDescription)")
          return
        }
        if Task.isCancelled { process.terminate() }

        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
          guard process.isRunning else { return }
          state.markTimedOut()
          process.terminate()
          DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
          }
        }
      }
    } onCancel: {
      if process.isRunning { process.terminate() }
    }
  }

  /// Coordinates the two pipe readers, the timeout, and the single continuation
  /// resume. Output is collected behind a lock because the readability handlers
  /// run on arbitrary queues.
  private final class ShellState: @unchecked Sendable {
    private let lock = NSLock()
    private var stdout = Data()
    private var stderr = Data()
    private var timedOut = false
    private var finished = false
    var continuation: CheckedContinuation<AgentToolOutput, Never>?

    func append(_ data: Data, isError: Bool) {
      guard !data.isEmpty else { return }
      lock.lock()
      if isError {
        stderr.append(data)
        if stderr.count > 2_000_000 { stderr = Data(stderr.suffix(1_000_000)) }
      } else {
        stdout.append(data)
        if stdout.count > 2_000_000 { stdout = Data(stdout.suffix(1_000_000)) }
      }
      lock.unlock()
    }

    func markTimedOut() {
      lock.lock()
      timedOut = true
      lock.unlock()
    }

    func finish(status: Int32) {
      lock.lock()
      guard !finished else {
        lock.unlock()
        return
      }
      finished = true
      let out = stdout
      let err = stderr
      let didTimeOut = timedOut
      let continuation = continuation
      self.continuation = nil
      lock.unlock()

      var text = String(decoding: out, as: UTF8.self)
      let errorText = String(decoding: err, as: UTF8.self)
      if !errorText.isEmpty {
        text += (text.isEmpty ? "" : "\n") + "[stderr]\n" + errorText
      }
      if text.isEmpty { text = "(no output)" }
      if didTimeOut { text += "\n[command timed out]" }
      text += "\n[exit code: \(status)]"
      continuation?.resume(
        returning: AgentToolOutput(
          text: truncateOutput(text), isError: status != 0 || didTimeOut))
    }

    func finish(error: String) {
      lock.lock()
      guard !finished else {
        lock.unlock()
        return
      }
      finished = true
      let continuation = continuation
      self.continuation = nil
      lock.unlock()
      continuation?.resume(returning: .failure(error))
    }
  }

  // MARK: - Helpers

  private static func requiredString(_ key: String, in arguments: [String: Any]) throws -> String {
    guard let value = optionalString(key, in: arguments), !value.isEmpty else {
      throw AgentToolError.invalidArguments("Missing required argument: \(key).")
    }
    return value
  }

  private static func optionalString(_ key: String, in arguments: [String: Any]) -> String? {
    arguments[key] as? String
  }

  private static func optionalInt(_ key: String, in arguments: [String: Any]) -> Int? {
    (arguments[key] as? NSNumber)?.intValue
  }

  private static func optionalBool(_ key: String, in arguments: [String: Any]) -> Bool? {
    (arguments[key] as? NSNumber)?.boolValue
  }

  private static func countOccurrences(of needle: String, in haystack: String) -> Int {
    var count = 0
    var start = haystack.startIndex
    while let range = haystack.range(of: needle, range: start..<haystack.endIndex) {
      count += 1
      start = range.upperBound
      if start == haystack.endIndex { break }
    }
    return count
  }

  /// Atomic writes replace the file, which drops its mode. Keep executable
  /// bits so an edit to a script doesn't silently disable it.
  private static func existingPermissions(at url: URL) -> Int? {
    guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else {
      return nil
    }
    return (attributes[.posixPermissions] as? NSNumber)?.intValue
  }

  private static func restore(_ permissions: Int?, to url: URL) {
    guard let permissions else { return }
    try? FileManager.default.setAttributes(
      [.posixPermissions: permissions], ofItemAtPath: url.path)
  }

  private static func truncateLine(_ line: String, limit: Int) -> String {
    guard line.count > limit else { return line }
    return String(line.prefix(limit)) + "…"
  }

  private static func preview(_ text: String, limit: Int) -> String {
    guard text.count > limit else { return text }
    return String(text.prefix(limit)) + "\n… [truncated]"
  }

  private static func truncateOutput(_ text: String, limit: Int = 30_000) -> String {
    guard text.count > limit else { return text }
    let head = 20_000
    return String(text.prefix(head)) + "\n… [output truncated] …\n" + String(text.suffix(limit - head))
  }

  /// Converts a glob into an anchored regular expression. `**/` matches any
  /// directory depth (including none), `**` matches anything, `*` stays within
  /// one path component, and `?` matches one non-slash character.
  static func globRegex(_ pattern: String) throws -> NSRegularExpression {
    let characters = Array(pattern)
    var regex = "^"
    var index = 0
    while index < characters.count {
      let character = characters[index]
      if character == "*" {
        if index + 1 < characters.count, characters[index + 1] == "*" {
          if index + 2 < characters.count, characters[index + 2] == "/" {
            regex += "(?:.*/)?"
            index += 3
          } else {
            regex += ".*"
            index += 2
          }
        } else {
          regex += "[^/]*"
          index += 1
        }
      } else if character == "?" {
        regex += "[^/]"
        index += 1
      } else {
        regex += NSRegularExpression.escapedPattern(for: String(character))
        index += 1
      }
    }
    regex += "$"
    do {
      return try NSRegularExpression(pattern: regex, options: [.caseInsensitive])
    } catch {
      throw AgentToolError.invalidArguments("Invalid glob pattern: \(pattern)")
    }
  }
}
