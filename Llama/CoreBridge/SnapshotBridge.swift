import Foundation

/// Reads immutable snapshots from the Rust core.
///
/// The core owns the logical state (`MIGRACAO_RUST_REVIEW.md` §3.4); Swift never
/// touches Rust memory, it asks for a JSON snapshot and decodes it. Today the
/// snapshot covers the `CoreConfig` the core holds (`core.snapshot`); as more
/// state migrates into the core, its snapshot grows here rather than the views
/// reading mutable Swift state directly.
enum SnapshotBridge {

  /// The raw `CoreSnapshot` from the core: `{ schemaVersion, config }`.
  static func snapshot() -> [String: Any]? {
    LlamaCore.snapshot()
  }

  /// The canonical `CoreConfig` the core currently holds (every key present).
  static func config() -> [String: Any]? {
    snapshot()?["config"] as? [String: Any]
  }

  /// The config schema version the core speaks.
  static func schemaVersion() -> Int? {
    (snapshot()?["schemaVersion"] as? NSNumber)?.intValue
  }
}
