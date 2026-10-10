import Foundation

/// Minimal reader for the GGUF key-value header.
///
/// GGUF files open with a small self-describing metadata section (magic,
/// version, tensor count, then typed key-value pairs) before the tensor data.
/// We parse just that section to answer questions the filename can't --
/// starting with embedded MTP heads, which some publishers (unsloth) ship
/// without any filename marker.
///
/// The parser lives in the Rust core (`gguf.*`, memory-mapped and fail-soft:
/// any structural surprise returns nil rather than throwing, so misreading a
/// model can never block install or scan). This type is a thin view so call
/// sites keep working unchanged.
enum GGUFMetadata {

  /// Whether the GGUF at `path` carries an embedded multi-token-prediction
  /// head, per its `<arch>.nextn_predict_layers` metadata key (the same key
  /// llama.cpp loads into `hparams.n_layer_nextn`). We match on the key
  /// suffix rather than resolving `general.architecture` first -- the key is
  /// namespaced per arch (`qwen35moe.nextn_predict_layers`, ...) but unique
  /// within a file. Returns nil when the header can't be parsed.
  static func hasEmbeddedMTPHead(path: String) -> Bool? {
    guard let out = LlamaCore.callJSON("gguf.hasEmbeddedMTPHead", ["path": path]) else {
      return nil
    }
    return out["value"] as? Bool
  }

  /// The model's native (training) context window, per its
  /// `<arch>.context_length` metadata key -- the value llama.cpp loads into
  /// `hparams.n_ctx_train`. Like the MTP head above we match on the key suffix
  /// rather than resolving `general.architecture` first, since the key is
  /// arch-namespaced (`gemma3.context_length`, ...) but unique within a file.
  ///
  /// The double optional keeps the two nil-ish outcomes apart, and callers care
  /// about the difference: outer nil = header unparseable, fall back to a
  /// default; `.some(nil)` = header fine but the key is genuinely absent, which
  /// means llama.cpp can't run the file at all. `llama-model.cpp` reads this key
  /// as required (unlike the optional hparams around it) for every arch but
  /// CLIP, so a clean absence marks a non-LLM GGUF -- e.g. speech-to-text models
  /// (`cohere_asr`, ...), which namespace their hparams under `stt.*` instead.
  static func contextLength(path: String) -> Int?? {
    guard let out = LlamaCore.callJSON("gguf.contextLength", ["path": path]) else {
      return nil
    }
    return .some(out["length"] as? Int)
  }

  /// Whether the GGUF at `path` is a decision model -- one that answers typed
  /// questions with a probability per option via llama-server's
  /// `/v1/systemone` endpoint, rather than generating text. llama.cpp keys this
  /// off `<arch>.decision.type` (`common.cpp`), so we do too: present means a
  /// decision model, whatever its value. Suffix-matched like the keys above.
  /// Returns nil when the header can't be parsed.
  static func isDecisionModel(path: String) -> Bool? {
    guard let out = LlamaCore.callJSON("gguf.isDecisionModel", ["path": path]) else {
      return nil
    }
    return out["value"] as? Bool
  }
}
