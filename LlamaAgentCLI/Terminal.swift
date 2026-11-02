import Darwin
import Foundation

enum Terminal {
  static var stdinIsTTY: Bool { isatty(STDIN_FILENO) == 1 }
  static var stdoutIsTTY: Bool { isatty(STDOUT_FILENO) == 1 }
  static var stderrIsTTY: Bool { isatty(STDERR_FILENO) == 1 }

  static func write(_ text: String, to handle: FileHandle = .standardOutput) {
    guard let data = text.data(using: .utf8) else { return }
    handle.write(data)
  }

  static func error(_ text: String) {
    write(text + "\n", to: .standardError)
  }

  static func bold(_ text: String) -> String {
    stdoutIsTTY ? "\u{001B}[1m\(text)\u{001B}[0m" : text
  }

  static func dim(_ text: String) -> String {
    stdoutIsTTY ? "\u{001B}[2m\(text)\u{001B}[0m" : text
  }

  static func cyan(_ text: String) -> String {
    stdoutIsTTY ? "\u{001B}[36m\(text)\u{001B}[0m" : text
  }

  static func red(_ text: String) -> String {
    stderrIsTTY ? "\u{001B}[31m\(text)\u{001B}[0m" : text
  }

  static func yellow(_ text: String) -> String {
    stderrIsTTY ? "\u{001B}[33m\(text)\u{001B}[0m" : text
  }

  static func readLine(prompt: String) -> String? {
    write(prompt, to: .standardError)
    return Swift.readLine()
  }

  static func truncate(_ text: String, limit: Int = 4_000) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count > limit else { return trimmed }
    return String(trimmed.prefix(limit)) + "\n… [truncated]"
  }
}

struct OutputRenderer {
  private var atLineStart = true

  mutating func text(_ piece: String) {
    guard !piece.isEmpty else { return }
    Terminal.write(piece)
    atLineStart = piece.last == "\n"
  }

  mutating func finishLine() {
    guard !atLineStart else { return }
    Terminal.write("\n")
    atLineStart = true
  }

  mutating func status(_ text: String) {
    finishLine()
    Terminal.error(text)
  }

  mutating func toolCall(title: String) {
    status(Terminal.cyan("→ \(title)"))
  }

  mutating func toolResult(_ text: String, isError: Bool) {
    finishLine()
    let preview = Terminal.truncate(text)
    let body = preview
      .split(separator: "\n", omittingEmptySubsequences: false)
      .map { "  \($0)" }
      .joined(separator: "\n")
    Terminal.error(isError ? Terminal.red(body) : Terminal.dim(body))
    atLineStart = true
  }
}
