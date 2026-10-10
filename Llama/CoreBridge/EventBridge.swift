import Foundation
import os.log

/// Marshals events emitted by the Rust core onto the main actor and re-posts
/// them as the app's existing `NotificationCenter` notifications, so moving
/// state ownership into the core (Fase 7) never changes the UI contract.
///
/// The core emits *semantic* event kinds (`serverState`, `modelStatus`, ...);
/// this bridge owns the kind → notification mapping, pinned by
/// `Fixtures/core_events.json` against the core's `events` module.
enum EventBridge {
  private static let logger = Logger(subsystem: Logging.subsystem, category: "EventBridge")

  /// Registers the C event callback. Call once at launch, after `ConfigBridge`.
  static func start() {
    LlamaCore.installEventCallback(receive)
  }

  /// The C-callable entry point. Must be a non-capturing `@convention(c)`
  /// function (stored as a `let` so its address is stable for the core to hold).
  static let receive: @convention(c) (UnsafePointer<CChar>?) -> Void = { ptr in
    guard let ptr, let json = String(validatingUTF8: ptr) else { return }
    dispatch(json: json)
  }

  /// Decodes an event payload and posts the mapped notification on the main
  /// actor. Unknown kinds are logged and ignored.
  static func dispatch(json: String) {
    guard
      let data = json.data(using: .utf8),
      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let kind = obj["event"] as? String
    else {
      logger.error("Malformed core event: \(json, privacy: .public)")
      return
    }
    guard let name = notification(forKind: kind) else {
      logger.debug("Ignoring unknown core event kind: \(kind, privacy: .public)")
      return
    }
    let userInfo = obj["payload"] as? [AnyHashable: Any]
    DispatchQueue.main.async {
      NotificationCenter.default.post(name: name, object: nil, userInfo: userInfo)
    }
  }

  /// Maps a semantic core event kind to the notification the UI already listens
  /// for. Mirrors `llama_core::events::notification_for_event`.
  static func notification(forKind kind: String) -> Notification.Name? {
    switch kind {
    case "serverState": return .LBServerStateDidChange
    case "modelDownloads": return .LBModelDownloadsDidChange
    case "modelDownloadedList": return .LBModelDownloadedListDidChange
    case "userSettings": return .LBUserSettingsDidChange
    case "modelDownloadFailed": return .LBModelDownloadDidFail
    case "modelDownloadCompleted": return .LBModelDownloadDidComplete
    case "modelStatus": return .LBModelStatusDidChange
    case "globalInputShortcut": return .LBGlobalInputShortcutDidChange
    default: return nil
    }
  }
}
