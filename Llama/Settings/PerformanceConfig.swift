import Foundation

/// How the app picks a context length for models without an explicit choice.
enum ContextLengthPolicy: String, CaseIterable, Identifiable, Codable {
  /// Use the largest tier the device can actually fit. This is the default:
  /// more context is the main reason to run a model locally.
  case largestFitting
  /// Keep the smallest usable context (4K) to leave as much memory free as
  /// possible.
  case minimum

  var id: String { rawValue }

  var label: String {
    switch self {
    case .largestFitting: "Largest that fits"
    case .minimum: "Smallest (4K)"
    }
  }
}

/// A KV cache element type understood by llama.cpp.
enum KVCacheType: String, CaseIterable, Identifiable, Codable {
  case f16
  case q8_0
  case q4_0
  case q4_1
  case q5_0
  case q5_1
  case iq4_nl

  var id: String { rawValue }
  var label: String { rawValue }
}

/// Memory-related llama.cpp preset keys. These are written into the global
/// `[*]` section of `models.ini`, with per-model overrides written into that
/// model's own section.
struct PerformanceConfig: Codable, Equatable {
  var flashAttention: Bool
  var kvCacheTypeK: KVCacheType
  var kvCacheTypeV: KVCacheType
  var parallel: Int
  var batchSize: Int
  var ubatchSize: Int
  var noKVOffload: Bool

  static let parallelRange = 1...8
  static let batchOptions = [128, 256, 512, 1024, 2048]
  static let ubatchOptions = [64, 128, 256, 512, 1024, 2048]

  /// More context and less memory: flash attention on, quantized KV cache, one
  /// slot, and conservative batches. Users can tune each value per model.
  static let `default` = PerformanceConfig(
    flashAttention: true,
    kvCacheTypeK: .q8_0,
    kvCacheTypeV: .q8_0,
    parallel: 1,
    batchSize: 512,
    ubatchSize: 256,
    noKVOffload: false
  )

  /// Clamps values that could otherwise make the preset invalid.
  var normalized: PerformanceConfig {
    var copy = self
    copy.parallel = min(max(copy.parallel, Self.parallelRange.lowerBound),
      Self.parallelRange.upperBound)
    copy.batchSize = Self.batchOptions.contains(copy.batchSize) ? copy.batchSize : 512
    copy.ubatchSize = Self.ubatchOptions.contains(copy.ubatchSize) ? copy.ubatchSize : 256
    if copy.ubatchSize > copy.batchSize {
      copy.ubatchSize = copy.batchSize
    }
    return copy
  }

  /// Preset keys, in a stable order. Values use the forms llama.cpp accepts in
  /// `models.ini` (`on`/`off` and `true`/`false`).
  var iniPairs: [(key: String, value: String)] {
    [
      ("flash-attn", flashAttention ? "on" : "off"),
      ("cache-type-k", kvCacheTypeK.rawValue),
      ("cache-type-v", kvCacheTypeV.rawValue),
      ("parallel", "\(parallel)"),
      ("batch-size", "\(batchSize)"),
      ("ubatch-size", "\(ubatchSize)"),
      ("no-kv-offload", noKVOffload ? "true" : "false"),
    ]
  }
}
