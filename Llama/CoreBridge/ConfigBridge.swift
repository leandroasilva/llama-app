import Foundation

/// Keeps the Rust core's `CoreConfig` snapshot in sync with the host
/// environment.
///
/// The core never reads `UserDefaults`; this bridge gathers the settings and
/// runtime measurements the core needs (cache location, port, memory, network
/// state) and pushes them via `llama_core_init` at launch and
/// `llama_core_set_config` on every `.LBUserSettingsDidChange`. The canonical
/// shape is defined by `CoreConfig` in `llama-core` and pinned by
/// `Fixtures/core_config.json`.
enum ConfigBridge {

  /// Pushes the current config and keeps it fresh on settings changes. Call once
  /// at launch, on the main thread, before the server starts.
  static func start() {
    LlamaCore.configure(currentConfig())
    NotificationCenter.default.addObserver(
      forName: .LBUserSettingsDidChange, object: nil, queue: .main
    ) { _ in
      push()
    }
  }

  /// Builds the `CoreConfig` dictionary from `UserSettings` plus runtime
  /// measurements. Only keys with a value are included; the core defaults the
  /// rest, so a partial push is valid.
  static func currentConfig() -> [String: Any] {
    let idle = UserSettings.sleepIdleTime.rawValue
    var cfg: [String: Any] = [
      "hfCacheDirectory": UserSettings.hfCacheDirectory.path,
      "serverPort": UserSettings.serverPort ?? 0,
      "agentMode": UserSettings.agentMode,
      "extraServerArgs": UserSettings.extraServerArgList,
      "gpuWorkingSetMb": Int(SystemMemory.gpuWorkingSetMb),
      "systemMemoryMb": Int(SystemMemory.memoryMb),
      "networkAvailable": currentNetworkAvailable(),
    ]
    if let token = UserSettings.hfToken { cfg["hfToken"] = token }
    // `sleepIdleTime` is -1 for "never unload"; the core models that as null.
    if idle >= 0 { cfg["sleepIdleSeconds"] = idle }
    if let webui = UserSettings.customWebUIDirectory {
      cfg["customWebUIDirectory"] = webui.path
    }
    return cfg
  }

  /// Pushes the current config to the core. Safe to call from the main thread.
  @discardableResult
  static func push() -> Int32 {
    LlamaCore.setConfig(currentConfig())
  }

  /// `ModelManager` is main-actor isolated and owns the `NWPathMonitor` status.
  /// Every call site runs on the main thread (launch, settings change, path
  /// update), so reading it here is isolated correctly.
  private static func currentNetworkAvailable() -> Bool {
    MainActor.assumeIsolated { ModelManager.shared.isNetworkAvailable }
  }
}
