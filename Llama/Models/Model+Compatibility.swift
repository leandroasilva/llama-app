import Foundation

extension Model {
  // MARK: - Memory Calculations

  /// We evaluate compatibility assuming a 4k-token context, which is the
  /// default llama.cpp launches with when no explicit value is provided.
  /// Models must also support at least this context length to launch.
  static let compatibilityCtxWindowTokens: Double = 4_096

  /// The memory a model may use on this device -- whichever of the two limits
  /// binds first:
  ///
  /// - `workingSet - fitSlack`: what the GPU will hold, less llama.cpp's
  ///   margin. We read Metal's working set rather than taking a fraction of
  ///   `hw.memsize` because Apple's ratio varies by machine (~75% on common
  ///   configurations, 84% measured on a 128 GB Mac) -- assuming 75%
  ///   understated large Macs and blocked sizes that run fine. llama.cpp
  ///   treats the same figure as the device's total and fits against it, so
  ///   this side matches `serve`'s own limit.
  /// - `physicalRam - osFloor`: leaves the desktop enough to work with. Only
  ///   binds on small Macs, where the memory outside the working set isn't
  ///   enough to host macOS on its own.
  ///
  /// With `useFullWorkingSet` set, the GPU limb is the working set itself --
  /// llama.cpp's fit slack is spent rather than held back. That's the whole
  /// override: it never exceeds what Metal recommends, so the machine is never
  /// pushed into swap by our arithmetic. Worth a gigabyte on every Mac, and a
  /// gigabyte is the difference for a model that lands just over the line.
  ///
  /// The floor stays fixed in both modes, which makes the flag inert on an
  /// 8 GB Mac (the floor is already what governs there). That's deliberate:
  /// taking a model past ~4 GB on an 8 GB machine doesn't just slow it down,
  /// it wedges it hard enough that the process can't be quit.
  ///
  /// The arithmetic lives in the Rust core (`compat.deviceBudget`).
  static var deviceMemoryBudgetMb: Double {
    LlamaCore.callValue("compat.deviceBudget", "mb", [
      "workingSetMb": Double(SystemMemory.gpuWorkingSetMb),
      "physicalMb": Double(SystemMemory.memoryMb),
      "useFullWorkingSet": UserSettings.useFullWorkingSet,
    ]) ?? 0
  }

  /// The margin to hand `serve` as `--fit-target`: everything between the
  /// working set and our budget, so its runtime fit lands on the same number
  /// our predictions do. Larger than the fit slack exactly when the OS floor is
  /// the binding limit.
  static var fitTargetMb: Double {
    LlamaCore.callValue("compat.fitTarget", "mb", [
      "workingSetMb": Double(SystemMemory.gpuWorkingSetMb),
      "budgetMb": deviceMemoryBudgetMb,
    ]) ?? 0
  }

  /// Rough pre-download fit check used by the deeplink resolver and the
  /// Discover picks: model weight memory ≈ fileSize × 1.05 must be within
  /// budget. Unknown sizes pass — don't filter out. The real compatibility
  /// check runs at launch once the MemProfile probe has measured resident bytes.
  static func estimatedWeightFits(bytes: Int64?, budgetMb: Double) -> Bool {
    var args: [String: Any] = ["budgetMb": budgetMb]
    if let bytes { args["bytes"] = bytes }
    return LlamaCore.callValue("compat.estimatedWeightFits", "value", args) ?? true
  }

  func isCompatible(
    ctxWindowTokens: Double = compatibilityCtxWindowTokens
  ) -> Bool {
    compatibilityInfo(ctxWindowTokens: ctxWindowTokens).isCompatible
  }

  func incompatibilitySummary(
    ctxWindowTokens: Double = compatibilityCtxWindowTokens
  ) -> String? {
    compatibilityInfo(ctxWindowTokens: ctxWindowTokens).incompatibilitySummary
  }

  func runtimeMemoryUsageMb(
    ctxWindowTokens: Double = compatibilityCtxWindowTokens
  ) -> UInt64 {
    LlamaCore.callValue("compat.runtimeMemoryMb", "mb", [
      "ctxWindow": ctxWindow,
      "fileSize": fileSize,
      "ctxBytesPer1kTokens": ctxBytesPer1kTokens,
      "residentBytes": residentBytes,
      "ctxWindowTokens": ctxWindowTokens,
    ]) ?? 0
  }

  // MARK: - Private Helpers

  /// Computes compatibility info for a model
  private func compatibilityInfo(
    ctxWindowTokens: Double = compatibilityCtxWindowTokens
  ) -> CompatibilityInfo {
    let out =
      LlamaCore.callJSON("compat.info", [
        "ctxWindow": ctxWindow,
        "fileSize": fileSize,
        "ctxBytesPer1kTokens": ctxBytesPer1kTokens,
        "residentBytes": residentBytes,
        "ctxWindowTokens": ctxWindowTokens,
        "budgetMb": Self.deviceMemoryBudgetMb,
      ]) ?? [:]
    return CompatibilityInfo(
      isCompatible: out["isCompatible"] as? Bool ?? false,
      incompatibilitySummary: out["incompatibilitySummary"] as? String
    )
  }

  /// Context tiers this device can actually run -- the ones the picker leaves
  /// selectable.
  /// When the MemProfile probe is still pending (or failed), returns only 4K as a safe
  /// default — without `ctxBytesPer1kTokens`, memory estimates are artificially low and
  /// all tiers would falsely appear compatible.
  var supportedContextTiers: [ContextTier] {
    let raws: [Int] =
      LlamaCore.callValue("compat.supportedTiers", "tiers", [
        "ctxWindow": ctxWindow,
        "fileSize": fileSize,
        "ctxBytesPer1kTokens": ctxBytesPer1kTokens,
        "residentBytes": residentBytes,
        "budgetMb": Self.deviceMemoryBudgetMb,
      ]) ?? [ContextTier.k4.rawValue]
    return raws.compactMap(ContextTier.init(rawValue:))
  }

  /// All tiers the model itself supports (within its native context window),
  /// regardless of whether this device has the memory to run them. The picker
  /// shows these, disabling the ones not in `supportedContextTiers`, so the
  /// user can see the model's full range rather than a silently truncated one.
  var displayContextTiers: [ContextTier] {
    let raws: [Int] =
      LlamaCore.callValue("compat.displayTiers", "tiers", [
        "ctxWindow": ctxWindow,
        "fileSize": fileSize,
        "ctxBytesPer1kTokens": ctxBytesPer1kTokens,
        "residentBytes": residentBytes,
      ]) ?? [ContextTier.k4.rawValue]
    return raws.compactMap(ContextTier.init(rawValue:))
  }

  /// The largest standard tier that fits within the model's native context window.
  /// Independent of device RAM -- this is the model's spec ceiling.
  var nativeMaxTier: ContextTier? {
    guard
      let raw: Int = LlamaCore.callValue(
        "compat.nativeMaxTier", "tier", ["ctxWindow": ctxWindow])
    else { return nil }
    return ContextTier(rawValue: raw)
  }

  /// The effective context tier for this model.
  /// Returns user's selection if set and still compatible, otherwise 4K.
  var effectiveCtxTier: ContextTier? {
    var args: [String: Any] = [
      "ctxWindow": ctxWindow,
      "fileSize": fileSize,
      "ctxBytesPer1kTokens": ctxBytesPer1kTokens,
      "residentBytes": residentBytes,
      "budgetMb": Self.deviceMemoryBudgetMb,
    ]
    if let selected = UserSettings.selectedCtxTier(for: id) {
      args["selected"] = selected.rawValue
    }
    guard let raw: Int = LlamaCore.callValue("compat.effectiveTier", "tier", args) else {
      return nil
    }
    return ContextTier(rawValue: raw)
  }

  private struct CompatibilityInfo {
    let isCompatible: Bool
    let incompatibilitySummary: String?
  }
}
