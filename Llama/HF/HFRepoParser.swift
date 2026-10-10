import Foundation

/// Parses HuggingFace repo directory names and GGUF filenames into structured
/// components. (Display parsing of a model's name, params and tags lives in
/// `ModelIdParser`, which works off the id and so covers every model alike.)
///
/// The parse lives in the Rust core (`repo.*`); this type is a thin view so
/// call sites keep working unchanged.
enum HFRepoParser {

  /// Splits a repo dir name (e.g. "models--bartowski--Llama-3.2-1B-Instruct-GGUF")
  /// into its org and repo. Returns nil if the format is unrecognized.
  static func parse(repoDir: String) -> (org: String, repo: String)? {
    guard let out = LlamaCore.callJSON("repo.parse", ["repoDir": repoDir]),
      let org = out["org"] as? String, let repo = out["repo"] as? String
    else { return nil }
    return (org, repo)
  }

  /// Extracts quantization from a GGUF filename.
  /// e.g. "Llama-3.2-1B-Instruct-Q4_K_M.gguf" → "Q4_K_M"
  /// e.g. "model-00001-of-00003.gguf" → nil (split shard, no quant in name)
  static func parseQuant(filename: String) -> String? {
    LlamaCore.callScalar("repo.parseQuant", ["filename": filename])
  }

  /// Returns true if the filename matches the split-shard pattern.
  /// e.g. "model-00001-of-00003.gguf" → true
  static func isSplitShard(_ filename: String) -> Bool {
    LlamaCore.callScalar("repo.isSplitShard", ["filename": filename]) ?? false
  }

  /// Returns true if this is the first shard of a split model.
  /// e.g. "model-00001-of-00003.gguf" → true
  /// e.g. "model-00002-of-00003.gguf" → false
  static func isFirstShard(_ filename: String) -> Bool {
    LlamaCore.callScalar("repo.isFirstShard", ["filename": filename]) ?? false
  }

  /// Returns the declared shard count of a split-shard filename.
  /// e.g. "model-00001-of-00003.gguf" → 3
  static func shardTotal(_ filename: String) -> Int? {
    LlamaCore.callScalar("repo.shardTotal", ["filename": filename])
  }

  /// Returns the base name for a split shard (everything before the shard pattern).
  /// e.g. "model-00001-of-00003.gguf" → "model"
  static func splitShardBaseName(_ filename: String) -> String? {
    LlamaCore.callScalar("repo.splitShardBaseName", ["filename": filename])
  }
}
