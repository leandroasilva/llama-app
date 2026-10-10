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

  /// Dispatches `method` and decodes a top-level JSON scalar (string, number,
  /// bool, or null) — methods like `repo.parseQuant` whose whole response is a
  /// bare value rather than an object.
  static func callScalar<T>(_ method: String, _ args: [String: Any] = [:]) -> T? {
    guard let raw = call(method, encode(args)), let data = raw.data(using: .utf8) else {
      return nil
    }
    return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) as? T
  }
}

