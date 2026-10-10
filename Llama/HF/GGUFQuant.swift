import Foundation

/// Quant naming lives in two forms, and this type owns both plus the boundary
/// between them:
///
/// - *label* — HF-style, keeps prefixes like Unsloth's `UD-` (`UD-Q4_K_XL`);
///   what filenames contain and what deeplink `quant=` params carry.
/// - *tag* — llama.cpp-canonical (`Q4_K_XL`); what model ids are built from.
///
/// The grammar (a byte-for-byte port of `parseGGUFQuantLabel` from
/// [huggingface.js/packages/tasks/src/gguf.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/gguf.ts))
/// lives in the Rust core (`quant.*`); this type is a thin view so call sites
/// keep working unchanged.
enum GGUFQuant {

  /// Returns the last quant label match in `path` (HF's `.at(-1)` semantics) or
  /// nil if none present. The returned string is uppercased and includes any
  /// `UD-` prefix and trailing `_<SUFFIX>` — exactly what HF would put in the
  /// deeplink's `quant=` param.
  static func parseLabel(_ path: String) -> String? {
    LlamaCore.callValue("quant.parseLabel", "label", ["path": path])
  }

  /// The one path → quant-tag derivation for building model ids. Every id
  /// construction site (the deeplink resolver and the post-install cache scan)
  /// must go through this: the two paths have to produce byte-identical ids or
  /// pending download rows never match their landed files and never get reaped.
  static func tag(forPath path: String, requested: String? = nil) -> String {
    var args: [String: Any] = ["path": path]
    if let requested { args["requested"] = requested }
    return LlamaCore.callValue("quant.tagForPath", "tag", args) ?? "unknown"
  }

  /// The quant's bit width, as llama.cpp's `extract_quant_bits`
  /// (`common/download.cpp`) derives it: take the file's trailing tag, then read
  /// the integer starting at its first digit — `Q4_K_M` → 4, `BF16` → 16,
  /// `F32` → 32, `NVFP4` → 4. Returns 0 when the path carries no tag.
  static func quantBits(forPath path: String) -> Int {
    LlamaCore.callValue("quant.quantBitsForPath", "bits", ["path": path]) ?? 0
  }

  /// `quantBits` for an already-isolated tag (`Q8_0` → 8).
  static func quantBits(forTag tag: String) -> Int {
    LlamaCore.callValue("quant.quantBitsForTag", "bits", ["tag": tag]) ?? 0
  }

  /// Convenience: true iff `a` and `b` parse to the same quant (case-insensitive).
  /// Used by `HFRepoResolver` to match URL-provided quants against siblings.
  static func matches(_ a: String, _ b: String) -> Bool {
    LlamaCore.callValue("quant.matches", "value", ["a": a, "b": b]) ?? false
  }

  /// Canonicalizes a quant tag the way llama.cpp does (`canonical_tag` in
  /// `common/preset.cpp`): keep only the last `-`/`.`-delimited token,
  /// uppercased — so `UD-Q4_K_XL` becomes `Q4_K_XL`.
  static func canonicalTag(_ label: String) -> String {
    LlamaCore.callValue("quant.canonicalTag", "tag", ["label": label]) ?? label.uppercased()
  }
}
