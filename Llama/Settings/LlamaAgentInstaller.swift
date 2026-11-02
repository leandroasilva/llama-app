import Foundation

/// Installs the `llama-agent` command that ships inside the app bundle. The
/// install is a symlink rather than a copy so Sparkle updates to Llama keep the
/// terminal command current without another install step.
enum LlamaAgentInstaller {
  static let installPath = "/usr/local/bin/llama-agent"

  enum InstallError: LocalizedError {
    case notBundled
    case failed(String)

    var errorDescription: String? {
      switch self {
      case .notBundled:
        return "This build doesn't include the llama-agent binary."
      case .failed(let message):
        return message
      }
    }
  }

  static var bundledURL: URL? {
    guard let url = Bundle.main.resourceURL?.appendingPathComponent("llama-agent"),
      FileManager.default.isExecutableFile(atPath: url.path)
    else { return nil }
    return url
  }

  static var isInstalled: Bool {
    FileManager.default.fileExists(atPath: installPath)
  }

  /// True when the installed command points at this app bundle's copy. A
  /// regular file (for example from `make install-agent`) reads as installed
  /// but not linked, so the button offers to replace it.
  static var isLinkedToBundle: Bool {
    guard let bundledURL else { return false }
    guard
      let destination = try? FileManager.default.destinationOfSymbolicLink(atPath: installPath)
    else { return false }
    return URL(fileURLWithPath: destination).standardizedFileURL.path
      == bundledURL.standardizedFileURL.path
  }

  static func install() async -> Result<String, Error> {
    guard let bundledURL else {
      return .failure(InstallError.notBundled)
    }

    let command =
      "/bin/mkdir -p /usr/local/bin && /bin/ln -sfn \(shellQuote(bundledURL.path)) "
      + "/usr/local/bin/llama-agent"
    let script = "do shell script \"\(appleScriptEscape(command))\" with administrator privileges"

    return await withCheckedContinuation { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
          try process.run()
          process.waitUntilExit()
        } catch {
          continuation.resume(returning: .failure(error))
          return
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)?
          .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        guard process.terminationStatus == 0 else {
          let message = output.isEmpty
            ? "The installer didn't finish." : output
          continuation.resume(returning: .failure(InstallError.failed(message)))
          return
        }

        guard isInstalled else {
          continuation.resume(
            returning: .failure(InstallError.failed("The command wasn't installed.")))
          return
        }
        continuation.resume(returning: .success(installPath))
      }
    }
  }

  private static func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }

  private static func appleScriptEscape(_ value: String) -> String {
    value.replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
  }
}
