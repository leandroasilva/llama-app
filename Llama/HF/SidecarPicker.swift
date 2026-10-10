import Foundation

/// Shared selection policy for sidecar GGUFs (`mmproj*.gguf` vision projectors
/// and the speculative draft heads named by `draftHeadPrefixes`).
///
/// Two discovery paths need the exact same policy — the deeplink resolver
/// choosing among HF API siblings (`HFRepoResolver`) and the cache scan
/// choosing among on-disk files (`HFCache`) — and the resulting attachment is
/// identity-relevant (it feeds `models.ini`), so the policy lives once instead
/// of being mirrored in both.
///
/// The selection itself is a port of llama.cpp's `find_best_sibling`
/// (`common/download.cpp`, the function behind its `find_best_mmproj` /
/// `find_best_mtp`) and lives in the Rust core (`sidecar.*`); this type is a
/// thin view so call sites keep working unchanged.
enum SidecarPicker {

  /// True for a vision-projector sidecar. The ggml-org convention is an
  /// `mmproj` prefix (`mmproj-*.gguf`), but community repos also use it as a
  /// suffix (e.g. `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`), so match `mmproj`
  /// anywhere in the basename. Accepts full repo-relative paths; only the
  /// basename is considered.
  static func isMmproj(_ path: String) -> Bool {
    LlamaCore.callValue("sidecar.isMmproj", "value", ["path": path]) ?? false
  }

  /// Filename prefixes llama.cpp resolves as draft heads shipped beside a model
  /// (`find_best_sibling` callers in `common/download.cpp`): `mtp-`
  /// (multi-token prediction), `dflash-` (block-diffusion drafting), `eagle3-`,
  /// and `dspark-`. Each borrows the target's token embeddings and output
  /// projection, so none loads as a model on its own.
  ///
  /// We drive `mtp-` and `dflash-` (see `DraftHead`). The rest are listed so
  /// main-model selection skips them too: a head parses to the target's quant
  /// tag, so an unrecognised one collides on `{org}/{repo}:{TAG}` and the
  /// dedupe can keep the head and drop the real model. Keep in step with upstream -- a scheme we don't list is a
  /// scheme that shadows real models.
  static let draftHeadPrefixes = ["mtp-", "dflash-", "eagle3-", "dspark-"]

  /// True for any draft-head sidecar (see `draftHeadPrefixes`). Accepts full
  /// repo-relative paths; only the basename is considered.
  static func isDraftHead(_ path: String) -> Bool {
    LlamaCore.callValue("sidecar.isDraftHead", "value", ["path": path]) ?? false
  }

  /// Picks the mmproj sidecar for `mainPath`. Repos routinely ship more than
  /// one (every current ggml-org VLM ships `mmproj-…-BF16` + `mmproj-…-Q8_0`),
  /// so ambiguity is the normal case, not a reason to skip: without an mmproj
  /// the model loads as text-only and image input silently disappears.
  static func mmproj(among names: [String], mainPath: String) -> String? {
    LlamaCore.callJSON("sidecar.mmproj", ["names": names, "mainPath": mainPath])?["path"]
      as? String
  }

  /// Picks the draft-head sidecar for `mainPath`, of the most preferred kind
  /// the repo ships (`DraftHead.preferenceOrder` -- DFlash before MTP). Kinds
  /// are tried one at a time, so a quant-matched MTP head never beats a DFlash
  /// head of a farther quant: the kind matters more than the quant. Within a
  /// kind, `tag` is the main file's canonical quant tag, which lets an exact
  /// `-<TAG>.gguf` head win over a merely near-in-bits one.
  ///
  /// This is where the app parts ways with `llama serve -hf`: plain `-hf`
  /// attaches no head unless asked for one by `--spec-type`. The kind of the
  /// returned path is `DraftHead(sidecarPath:)`.
  static func draftHead(among names: [String], mainPath: String, tag: String?) -> String? {
    var args: [String: Any] = ["names": names, "mainPath": mainPath]
    if let tag { args["tag"] = tag }
    return LlamaCore.callJSON("sidecar.draftHead", args)?["path"] as? String
  }
}
