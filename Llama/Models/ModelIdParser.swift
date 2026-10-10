import Foundation

/// Presentation-only parse of a model id into a short name plus metadata
/// chips, mirroring the llama.cpp WebUI's default model rendering
/// (`tools/ui` `parseModelId`) so a model looks the same in our menu and in
/// the WebUI's picker. The id itself is never touched — every field here is
/// derived from the id string, so display can never disagree with what
/// `curl /v1/models` returns.
///
/// The parse lives in the Rust core (`modelId.parse`); this type is a thin
/// view so call sites keep working unchanged.
enum ModelIdParser {

  /// The parsed components of a model id, ready for rendering.
  struct Parsed {
    /// The id's pre-slash org, verbatim (e.g. "unsloth"). Nil for ids with
    /// no slash. Callers render `displayOrg`, not this.
    let org: String?
    /// The org as a row renders it: nil for the default org (`ggml-org`),
    /// the org verbatim for everything else. Computed by the core so the
    /// default-org rule lives in one place.
    let displayOrg: String?
    /// Repo-name segments before the params segment, rejoined with dashes
    /// (e.g. "gemma-3", "Devstral-Small-2"). Falls back to the whole repo
    /// name (minus GGUF/GGML segments) when no params segment is found.
    let name: String
    /// Parameter-count chip, uppercased (e.g. "24B", "270M"); for MoE repos
    /// that name their activated params it's the combined form ("35B-A3B").
    let params: String?
    /// The id's post-colon quant tag, verbatim (e.g. "Q4_K_M"). Nil when the
    /// id has no colon (catalog rows before a quant is known) or the tag is
    /// the "unknown" fallback (unlabeled files).
    let quant: String?
    /// Leftover repo segments after the params segment (e.g. "it", "qat",
    /// "instruct", "2512"). Empty when no params segment is found — the
    /// leftovers stay in the name then, matching the WebUI.
    let tags: [String]
  }

  /// What a row visibly renders when tags are hidden: org prefix + name +
  /// params + quant. Two ids with equal keys would look identical in a list,
  /// so callers use key collisions to decide when a row needs its tags shown
  /// after all. The displayed org (not the raw one) is what's included, so
  /// two same-named repos from different orgs don't count as colliding — the
  /// `org/` prefix already tells them apart.
  static func displayKey(_ id: String) -> String {
    LlamaCore.callValue("modelId.displayKey", "key", ["id": id]) ?? ""
  }

  /// The row rendering as one plain string, e.g. `unsloth/Qwen3-Coder 30B-A3B
  /// Q4_K_M` -- for places that can't draw chips, like a SwiftUI picker's
  /// menu. Same parts in the same order as a row, so a model reads the same.
  static func plainName(_ id: String, showTags: Bool) -> String {
    LlamaCore.callValue("modelId.plainName", "name", ["id": id, "showTags": showTags]) ?? id
  }

  static func parse(_ id: String) -> Parsed {
    let out = LlamaCore.callJSON("modelId.parse", ["id": id]) ?? [:]
    return Parsed(
      org: out["org"] as? String,
      displayOrg: out["displayOrg"] as? String,
      name: out["name"] as? String ?? id,
      params: out["params"] as? String,
      quant: out["quant"] as? String,
      tags: out["tags"] as? [String] ?? []
    )
  }
}
