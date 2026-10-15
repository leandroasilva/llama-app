import Foundation

/// Thin Swift façade over the Rust core's C ABI.
///
/// This is the single place the app talks to `llama-core-ffi`. It owns the
/// memory rule of the boundary: the core returns a heap JSON string that must
/// be released with `llama_core_free`, which `call` does via `defer`.
enum LlamaCore {

  /// Dispatches a method to the Rust core with a JSON `args` object. Returns the
  /// JSON response, or `nil` if the core reported a failure (e.g. a panic).
  static func call(_ method: String, _ args: String = "{}") -> String? {
    var result: String?
    method.withCString { m in
      args.withCString { a in
        guard let raw = llama_core_call(m, a) else { return }
        defer { llama_core_free(raw) }
        result = String(cString: raw)
      }
    }
    return result
  }

  /// Encodes a Swift dictionary as the JSON `args` object for `call`.
  static func encode(_ args: [String: Any]) -> String {
    guard
      let data = try? JSONSerialization.data(withJSONObject: args),
      let json = String(data: data, encoding: .utf8)
    else { return "{}" }
    return json
  }

  /// Dispatches `method` and decodes an object response. Returns `nil` when the
  /// core returned NULL or a non-object payload.
  static func callJSON(_ method: String, _ args: [String: Any] = [:]) -> [String: Any]? {
    guard let raw = call(method, encode(args)), let data = raw.data(using: .utf8) else {
      return nil
    }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
  }

  /// Dispatches `method` and decodes an array response (e.g. `contextTier.list`).
  static func callJSONArray(_ method: String, _ args: [String: Any] = [:]) -> [Any]? {
    guard let raw = call(method, encode(args)), let data = raw.data(using: .utf8) else {
      return nil
    }
    return (try? JSONSerialization.jsonObject(with: data)) as? [Any]
  }

  /// Convenience for methods that return a single top-level value under `key`.
  static func callValue<T>(_ method: String, _ key: String, _ args: [String: Any] = [:]) -> T? {
    callJSON(method, args)?[key] as? T
  }

  /// Streams raw bytes to a `.partial` writer opened via `partial.open`.
  ///
  /// The one write-side call that bypasses the JSON bridge: download bytes are
  /// arbitrary binary (GGUF is not UTF-8) and can't ride a NUL-terminated C
  /// string. Returns the core's status code — 0 on success.
  static func partialAppend(handle: UInt64, data: Data) -> Int32 {
    data.withUnsafeBytes { raw -> Int32 in
      let buffer = raw.bindMemory(to: UInt8.self)
      return llama_core_partial_append(handle, buffer.baseAddress, buffer.count)
    }
  }

  /// Dispatches `method` and decodes a top-level JSON scalar (string, number,
  /// bool, or null) — methods like `repo.parseQuant` whose whole response is a
  /// bare value rather than an object.
  static func callScalar<T>(_ method: String, _ args: [String: Any] = [:]) -> T? {
    guard let raw = call(method, encode(args)), let data = raw.data(using: .utf8) else {
      return nil
    }
    return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) as? T
  }

  // MARK: - Config / lifecycle

  /// Initializes the core with a `CoreConfig` snapshot (see `ConfigBridge`).
  /// Returns the core's status code (0 on success).
  @discardableResult
  static func configure(_ config: [String: Any]) -> Int32 {
    encode(config).withCString { llama_core_init($0) }
  }

  /// Replaces the core's `CoreConfig` snapshot on a settings change.
  @discardableResult
  static func setConfig(_ config: [String: Any]) -> Int32 {
    encode(config).withCString { llama_core_set_config($0) }
  }

  /// Registers (or clears, with `nil`) the event callback. The callback fires on
  /// whatever thread the core emits from — `EventBridge` marshals to the main
  /// actor.
  static func installEventCallback(
    _ cb: (@convention(c) (UnsafePointer<CChar>?) -> Void)?
  ) {
    llama_core_set_event_cb(cb)
  }

  /// Returns the core's canonical `CoreConfig` snapshot (every key present).
  static func snapshot() -> [String: Any]? {
    callJSON("core.snapshot")
  }
}

