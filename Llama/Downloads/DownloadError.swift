import Foundation

/// Describes errors that can occur before a model download begins. The
/// user-facing wording is produced by the Rust core (`download.errorMessage`).
enum DownloadError: Error, LocalizedError {
  case notCompatible(reason: String)
  case notEnoughDiskSpace(required: String, available: String)

  private var arguments: [String: String] {
    switch self {
    case .notCompatible(let reason):
      return ["kind": "notCompatible", "reason": reason]
    case .notEnoughDiskSpace(let required, let available):
      return ["kind": "notEnoughDiskSpace", "required": required, "available": available]
    }
  }

  var errorDescription: String? {
    LlamaCore.callValue("download.errorMessage", "description", arguments)
  }

  var recoverySuggestion: String? {
    LlamaCore.callValue("download.errorMessage", "recovery", arguments)
  }
}
