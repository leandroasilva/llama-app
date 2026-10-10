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
}
