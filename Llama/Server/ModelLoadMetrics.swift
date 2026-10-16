import Darwin
import Foundation
import os.log

/// One completed model-load observation. This is deliberately local and
/// aggregate-only: it records timing and process memory, never prompts or
/// generated text.
struct ModelLoadMetric: Codable {
  let modelId: String
  let startedAt: Date
  let durationMs: Int
  let contextTokens: Int
  let fileSizeBytes: Int64
  let hasMmproj: Bool
  let hasDraft: Bool
  let sampleCount: Int
  let peakResidentBytes: UInt64
  let peakPhysicalFootprintBytes: UInt64
  let finalResidentBytes: UInt64
  let finalPhysicalFootprintBytes: UInt64
  let succeeded: Bool
}

/// Collects load timing and child-process memory without changing the server's
/// loading policy. The server already polls model state once per second while a
/// load is active, so sampling is attached to that existing cadence.
@MainActor
final class ModelLoadMetrics {
  private struct ActiveLoad {
    let modelId: String
    let startedAt: Date
    let contextTokens: Int
    let fileSizeBytes: Int64
    let hasMmproj: Bool
    let hasDraft: Bool
    var sampleCount = 0
    var peakResidentBytes: UInt64 = 0
    var peakPhysicalFootprintBytes: UInt64 = 0
    var lastSnapshot = ProcessMemorySnapshot.zero
  }

  private static let logger = Logger(subsystem: Logging.subsystem, category: "ModelLoadMetrics")
  private static let maxRecords = 100

  private var active: ActiveLoad?
  private var records: [ModelLoadMetric] = []

  init() {
    records = Self.loadRecords()
  }

  var activeModelId: String? {
    active?.modelId
  }

  func begin(model: Model, processID: pid_t?) {
    guard active == nil else { return }

    active = ActiveLoad(
      modelId: model.id,
      startedAt: Date(),
      contextTokens: model.effectiveCtxTier?.rawValue ?? 4_096,
      fileSizeBytes: model.fileSize,
      hasMmproj: model.mmprojUrl != nil,
      hasDraft: model.draftHead != nil
    )
    sample(processID: processID)
  }

  func sample(processID: pid_t?) {
    guard var active else { return }
    let snapshot = processID.flatMap(ProcessMemorySnapshot.init(pid:)) ?? .zero
    active.sampleCount += 1
    active.lastSnapshot = snapshot
    active.peakResidentBytes = max(active.peakResidentBytes, snapshot.residentBytes)
    active.peakPhysicalFootprintBytes = max(
      active.peakPhysicalFootprintBytes, snapshot.physicalFootprintBytes)
    self.active = active
  }

  func finish(succeeded: Bool, processID: pid_t?) {
    guard var active else { return }
    sample(processID: processID)
    guard let sampled = self.active else { return }
    active = sampled

    let metric = ModelLoadMetric(
      modelId: active.modelId,
      startedAt: active.startedAt,
      durationMs: max(0, Int(Date().timeIntervalSince(active.startedAt) * 1_000)),
      contextTokens: active.contextTokens,
      fileSizeBytes: active.fileSizeBytes,
      hasMmproj: active.hasMmproj,
      hasDraft: active.hasDraft,
      sampleCount: active.sampleCount,
      peakResidentBytes: active.peakResidentBytes,
      peakPhysicalFootprintBytes: active.peakPhysicalFootprintBytes,
      finalResidentBytes: active.lastSnapshot.residentBytes,
      finalPhysicalFootprintBytes: active.lastSnapshot.physicalFootprintBytes,
      succeeded: succeeded
    )

    records.append(metric)
    if records.count > Self.maxRecords {
      records.removeFirst(records.count - Self.maxRecords)
    }
    self.active = nil
    persist()

    Self.logger.info(
      "Model load \(metric.succeeded ? "completed" : "failed", privacy: .public): \(metric.modelId, privacy: .public), duration=\(metric.durationMs)ms, context=\(metric.contextTokens), peakRSS=\(metric.peakResidentBytes / 1_048_576)MiB, peakFootprint=\(metric.peakPhysicalFootprintBytes / 1_048_576)MiB, samples=\(metric.sampleCount)"
    )
  }

  private static var recordsURL: URL {
    UserSettings.appSupportDir
      .appendingPathComponent("metrics", isDirectory: true)
      .appendingPathComponent("model-loads.json")
  }

  private static func loadRecords() -> [ModelLoadMetric] {
    guard let data = try? Data(contentsOf: recordsURL),
      let records = try? JSONDecoder().decode([ModelLoadMetric].self, from: data)
    else { return [] }
    return Array(records.suffix(maxRecords))
  }

  private func persist() {
    do {
      let directory = Self.recordsURL.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      let data = try JSONEncoder().encode(records)
      try data.write(to: Self.recordsURL, options: .atomic)
    } catch {
      Self.logger.error("Failed to persist model-load metrics: \(error.localizedDescription)")
    }
  }
}

private struct ProcessMemorySnapshot {
  let residentBytes: UInt64
  let physicalFootprintBytes: UInt64

  private init(residentBytes: UInt64, physicalFootprintBytes: UInt64) {
    self.residentBytes = residentBytes
    self.physicalFootprintBytes = physicalFootprintBytes
  }

  static let zero = ProcessMemorySnapshot(residentBytes: 0, physicalFootprintBytes: 0)

  init?(pid: pid_t) {
    var taskInfo = proc_taskinfo()
    let taskInfoSize = Int32(MemoryLayout<proc_taskinfo>.size)
    let result = withUnsafeMutableBytes(of: &taskInfo) {
      proc_pidinfo(pid, PROC_PIDTASKINFO, 0, $0.baseAddress, taskInfoSize)
    }
    guard result == taskInfoSize else { return nil }

    residentBytes = taskInfo.pti_resident_size
    physicalFootprintBytes = Self.physicalFootprint(pid: pid) ?? residentBytes
  }

  private static func physicalFootprint(pid: pid_t) -> UInt64? {
    var info = rusage_info_v4()
    let result = withUnsafeMutableBytes(of: &info) {
      proc_pid_rusage(
        pid, RUSAGE_INFO_V4,
        $0.baseAddress?.assumingMemoryBound(to: rusage_info_t?.self))
    }
    guard result == 0 else { return nil }
    return info.ri_phys_footprint
  }
}
