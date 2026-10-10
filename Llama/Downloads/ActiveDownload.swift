import Foundation

/// Paused download state. Carries the `Model` so deeplink-sideload
/// placeholders can still render a paused row after their active download
/// is torn down.
struct PausedDownload {
  let model: Model
  let bytesOnDisk: Int64
  /// True when the pause was forced by the network path dropping mid-download;
  /// such entries auto-resume on the next offline→online edge (`handlePathUpdate`).
  /// User-initiated pauses keep this false so they stay paused. In-memory only —
  /// a restart rehydrates paused rows from placeholders with the flag off.
  var resumeOnReconnect = false
}

/// Tracks the progress of a multi-file model download.
///
/// Bytes-in-flight are tracked externally (ModelManager owns the per-task
/// `PartialWriter` state since we stream into `.partial` files ourselves now,
/// rather than relying on URLSession's `countOfBytesReceived`). The caller
/// passes current sums into `refreshProgress`.
struct ActiveDownload {
  let model: Model
  var progress: Progress
  var tasks: [Int: URLSessionDataTask]
  /// Bytes belonging to files that have already completed (hash-verified and promoted into HF cache).
  var completedFilesBytes: Int64 = 0
  /// HF download plan (commit hash + blob hashes) needed to write into the HF
  /// cache layout. Nil between placeholder creation and the async metadata
  /// fetch completing. Shares this entry's lifecycle, so it can't drift out of
  /// sync with the download the way a parallel dict would.
  var plan: HFDownloadPlan?
  /// Retry attempts consumed per file URL (exponential-backoff budget).
  /// Living here, the counters die with the entry — teardown and completion
  /// need no separate cleanup, and a later resume starts a fresh budget.
  var retryAttempts: [URL: Int] = [:]
  /// When the last progress notification was posted (UI refresh throttle).
  var lastNotified: Date = .distantPast
  /// Recent (time, completed-bytes) samples backing the live rate readout.
  /// Appended on every `refreshProgress` tick, pruned to `rateWindow`. A
  /// trailing-window average keeps the displayed figure lively but not jumpy.
  var rateSamples: [(time: Date, bytes: Int64)] = []

  /// How far back the rate average looks.
  private static let rateWindow: TimeInterval = 5

  /// Current transfer rate in bytes/sec, averaged over `rateWindow`.
  /// Nil until the window holds at least a second of data — a shorter span
  /// makes the first readouts wildly noisy. Computed by the Rust core
  /// (`download.rate`), which also clamps a rewound byte count to zero.
  var bytesPerSecond: Int64? {
    let samples: [[Any]] = rateSamples.map {
      [$0.time.timeIntervalSince1970, Int($0.bytes)]
    }
    return (LlamaCore.callJSON("download.rate", ["samples": samples, "window": Self.rateWindow])?["rate"]
      as? NSNumber)?.int64Value
  }

  mutating func addTask(_ task: URLSessionDataTask) {
    tasks[task.taskIdentifier] = task
  }

  mutating func removeTask(with identifier: Int) {
    tasks.removeValue(forKey: identifier)
  }

  mutating func markTaskFinished(_ task: URLSessionDataTask, fileSize: Int64) {
    tasks.removeValue(forKey: task.taskIdentifier)
    completedFilesBytes += fileSize
  }

  /// Refreshes `progress` from caller-supplied byte sums across active tasks.
  /// `activeBytes` is the total bytes currently on disk across all in-flight `.partial` files;
  /// `expectedActiveBytes` is the sum of each task's known total size (0 before the response arrives).
  mutating func refreshProgress(activeBytes: Int64, expectedActiveBytes: Int64) {
    // The progress math (don't shrink totalUnitCount, `max(…, 1)` floor) is
    // done by the Rust core (`download.refreshProgress`).
    let out = LlamaCore.callJSON("download.refreshProgress", [
      "totalUnitCount": progress.totalUnitCount,
      "completedFilesBytes": completedFilesBytes,
      "activeBytes": activeBytes,
      "expectedActiveBytes": expectedActiveBytes,
    ])
    progress.totalUnitCount =
      (out?["totalUnitCount"] as? NSNumber)?.int64Value ?? max(progress.totalUnitCount, 1)
    progress.completedUnitCount =
      (out?["completedUnitCount"] as? NSNumber)?.int64Value ?? (completedFilesBytes + activeBytes)

    // Feed the rate window (pruning stays here; only the average is in Rust).
    let now = Date()
    rateSamples.append((time: now, bytes: completedFilesBytes + activeBytes))
    rateSamples.removeAll { now.timeIntervalSince($0.time) > Self.rateWindow }
  }

  var isEmpty: Bool { tasks.isEmpty }
}
