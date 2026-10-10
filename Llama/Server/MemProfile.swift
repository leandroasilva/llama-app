import Foundation
import os.log

/// Per-model memory profile: parameters of an affine estimator that predicts
/// total memory at any context size. Currently produced by probing with
/// `llama fit-params`, but the data is independent of how it's measured.
/// Cached to disk so we don't re-run on every launch.
///
/// All fields are non-optional on purpose: when we add a new field, synthesized
/// Codable fails to decode old cache entries (keyNotFound), `MemProfileCache.get`
/// returns nil, and we re-probe to produce a fresh entry. Don't switch to
/// `decodeIfPresent` or a custom `init(from:)` with defaults — that would
/// silently keep stale entries across upgrades.
struct MemProfile: Codable {
  /// Slope of the affine memory model, in bytes per 1k tokens.
  ///   mem(ctx) = residentBytes + ctxBytesPer1kTokens · ctx / 1000
  /// Maps directly to Model.ctxBytesPer1kTokens.
  let ctxBytesPer1kTokens: Int
  /// Intercept of the affine memory model, in bytes. Total footprint at ctx=0 —
  /// includes model weights, compute buffers, and any ctx-independent KV state
  /// (e.g. the per-layer local cache that SWA models like Gemma keep regardless
  /// of ctx). Maps to Model.residentBytes.
  /// 0 if unknown (e.g. probe failed).
  let residentBytes: Int
  /// Schema version. Bumping invalidates on-disk caches via the Codable
  /// keyNotFound mechanism documented at the top of this file.
  /// v2: switched from single-probe debug-table parse to two-probe -fitp affine
  ///     fit. Semantics of both above fields changed — v1 caches are discarded.
  let schemaVersion: Int

  init(ctxBytesPer1kTokens: Int, residentBytes: Int = 0) {
    self.ctxBytesPer1kTokens = ctxBytesPer1kTokens
    self.residentBytes = residentBytes
    self.schemaVersion = 2
  }
}

/// Probes a model with `llama fit-params` to derive a `MemProfile`.
/// Run once per installed model; the result is cached on disk.
///
/// The subprocess plumbing lives here (Swift owns `Process` and cancellation);
/// the affine fit and the `-fitp` output parse live in the Rust core
/// (`engine.fit` / `engine.parseFitp`).
enum MemProfileRunner {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "MemProfileRunner")

  /// Probes a model at two context sizes and fits an affine memory model
  ///   mem(ctx) = a + b·ctx
  /// where `a` becomes residentBytes and `b·1000` becomes ctxBytesPer1kTokens.
  ///
  /// Why two probes, not one: memory is not purely linear in ctx. SWA models
  /// (Gemma family) keep a fixed-size per-layer local KV cache regardless of
  /// ctx, which shows up as a non-zero intercept (~174 MiB on Gemma 3 4B).
  /// Dividing a single 128k probe by 128 folds that intercept into the slope
  /// and mis-predicts small-ctx usage by multiple hundreds of MiB. A two-point
  /// affine fit recovers both terms correctly and degenerates cleanly to
  /// near-linear for dense models (where a ≈ 0).
  ///
  /// Takes ~2s per model (two probes). Returns nil on failure.
  /// Supports cancellation — terminates the subprocess if the Task is cancelled.
  static func run(modelPath: String) async -> MemProfile? {
    guard let llamaPath = LlamaBinaries.llamaPath else {
      logger.error("llama binary not found")
      return nil
    }

    let ctxLo: UInt32 = 4096
    let ctxHi: UInt32 = 131072

    guard let loMib = await probeTotalMib(llama: llamaPath, modelPath: modelPath, ctx: ctxLo)
    else { return nil }
    guard let hiMib = await probeTotalMib(llama: llamaPath, modelPath: modelPath, ctx: ctxHi)
    else { return nil }

    // Affine fit in the core: solves a + b·ctx_lo = lo, a + b·ctx_hi = hi, and
    // rejects a non-monotonic total (negative slope) as a broken probe, which
    // routes the caller to the file-size fallback instead of a bogus
    // "memory is ctx-independent" estimate that could OOM at runtime.
    guard
      let out = LlamaCore.callJSON("engine.fit", [
        "loMib": Int(loMib), "hiMib": Int(hiMib), "ctxLo": Int(ctxLo), "ctxHi": Int(ctxHi),
      ]),
      let profile = out["profile"] as? [String: Any],
      let bBytesPer1k = (profile["ctxBytesPer1kTokens"] as? NSNumber)?.intValue,
      let aBytes = (profile["residentBytes"] as? NSNumber)?.intValue
    else {
      logger.error(
        "Fit params: non-monotonic total (\(loMib) MiB at \(ctxLo) vs \(hiMib) MiB at \(ctxHi))")
      return nil
    }

    logger.info(
      "Mem profile (affine): total(\(ctxLo))=\(loMib) MiB, total(\(ctxHi))=\(hiMib) MiB → a=\(aBytes) bytes, b=\(bBytesPer1k) bytes/1k"
    )

    return MemProfile(ctxBytesPer1kTokens: bBytesPer1k, residentBytes: aBytes)
  }

  /// Runs `llama fit-params -c <ctx> -fitp on` and returns the total MiB
  /// footprint across all devices (sum of model + context + compute columns).
  ///
  /// `-fitp on` writes one line per device to stdout, e.g.:
  ///   MTL0 2402 2734 517
  ///   Host 680 0 538
  /// The core sums every numeric column across every row — on Apple Silicon
  /// unified memory every row draws from the same physical RAM, so all rows
  /// count toward the same budget.
  private static func probeTotalMib(
    llama: String, modelPath: String, ctx: UInt32
  ) async -> Int? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: llama)
    process.arguments = ["fit-params", "-m", modelPath, "-c", String(ctx), "-fitp", "on"]

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    do {
      try process.run()
    } catch {
      logger.error("Failed to launch llama fit-params: \(error.localizedDescription)")
      return nil
    }

    return await withTaskCancellationHandler {
      // Read both pipes concurrently to avoid buffer-fill deadlocks. stderr
      // receives heavy log spam from the model load; we discard it.
      async let stdoutRead = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
      async let stderrRead = stderrPipe.fileHandleForReading.readDataToEndOfFile()

      let (stdout, stderr) = await (stdoutRead, stderrRead)
      process.waitUntilExit()

      guard process.terminationStatus == 0 else {
        let errOutput = String(decoding: stderr, as: UTF8.self)
        // status 15 = SIGTERM from our cancellation handler, don't log
        if process.terminationStatus != 15 {
          logger.error(
            "llama fit-params (ctx=\(ctx)) exited with status \(process.terminationStatus): \(errOutput.prefix(500))"
          )
        }
        return nil
      }

      let output = String(decoding: stdout, as: UTF8.self)

      // The core parses the `-fitp` rows: each is a non-numeric device name
      // followed by three ints (model/context/compute), summed across rows.
      // A response of `null` means no usable row was found.
      let out = LlamaCore.callJSON("engine.parseFitp", ["output": output])
      guard let total = (out?["sum"] as? NSNumber)?.intValue else {
        logger.error("Failed to parse -fitp output for ctx=\(ctx): \(output.prefix(500))")
        return nil
      }
      return total
    } onCancel: {
      if process.isRunning {
        process.terminate()
      }
    }
  }
}

/// Disk cache for `MemProfile` results.
/// Stored in ~/Library/Caches/{bundleId}/MemProfile/ — the macOS-standard
/// location for derived, recreatable data. If the system purges the cache, we
/// re-probe.
///
/// The file naming (SHA-256 of the id) and the JSON write/read live in the Rust
/// core (`engine.memCacheGet` / `engine.memCacheSet`); this type supplies the
/// platform cache directory.
enum MemProfileCache {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "MemProfileCache")

  /// Returns the cached MemProfile for a model ID, or nil if not cached.
  static func get(modelId: String) -> MemProfile? {
    guard
      let out = LlamaCore.callJSON("engine.memCacheGet", [
        "cacheDir": cacheDir.path, "modelId": modelId,
      ]),
      let profile = out["profile"] as? [String: Any],
      let ctxBytesPer1kTokens = (profile["ctxBytesPer1kTokens"] as? NSNumber)?.intValue,
      let residentBytes = (profile["residentBytes"] as? NSNumber)?.intValue
    else { return nil }
    return MemProfile(ctxBytesPer1kTokens: ctxBytesPer1kTokens, residentBytes: residentBytes)
  }

  /// Stores a MemProfile for a model ID.
  static func set(_ profile: MemProfile, for modelId: String) {
    let out = LlamaCore.callJSON("engine.memCacheSet", [
      "cacheDir": cacheDir.path,
      "modelId": modelId,
      "ctxBytesPer1kTokens": profile.ctxBytesPer1kTokens,
      "residentBytes": profile.residentBytes,
    ])
    if let error = out?["error"] as? String {
      logger.error("Failed to cache mem profile for \(modelId): \(error)")
    }
  }

  // MARK: - Private

  private static var cacheDir: URL {
    let bundleId = Bundle.main.bundleIdentifier ?? "app.llama.Llama"
    return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
      .appendingPathComponent(bundleId)
      .appendingPathComponent("MemProfile")
  }
}
