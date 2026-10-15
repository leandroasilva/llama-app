import Foundation

// MARK: - WriterTable

/// Thread-safe table of in-flight `PartialWriter`s, keyed by `URLSessionTask.taskIdentifier`.
///
/// The streaming state itself (file handle, running SHA256, byte counter) lives
/// in the Rust core behind `partial.*`; this table is only the per-task
/// association URLSession needs, so it stays in Swift. Owns the serial queue
/// that guards both the dictionary and the mirrored counters of the writers it
/// holds: a `PartialWriter`'s mutable fields are only ever touched inside
/// `sync`, so the writers themselves need no further synchronization.
/// Encapsulating the dict + queue here gives the shared mutable state a single
/// owner, instead of a raw `nonisolated(unsafe)` field living on the otherwise
/// main-actor `ModelManager` — hence `@unchecked Sendable`.
final class WriterTable: @unchecked Sendable {
  private var writers: [Int: PartialWriter] = [:]
  private let queue = DispatchQueue(
    label: "app.llama.ModelManager.writers", qos: .userInitiated)

  /// Runs `body` with exclusive access to the writer table. The sole entry
  /// point — every read or mutation of a writer goes through here.
  func sync<T>(_ body: (inout [Int: PartialWriter]) -> T) -> T {
    queue.sync { body(&writers) }
  }
}

// MARK: - PartialWriter

/// Per-file download state: the Rust core's streaming writer handle, plus the
/// immutable download context (model, cache dir, HF plan) the delegate callbacks
/// need.
///
/// The file handle and running SHA256 live in the core, not here: GGUF bytes are
/// not UTF-8, so the write path goes through `llama_core_partial_append` (a raw
/// byte escape hatch beside the JSON `partial.*` ops) rather than the JSON
/// bridge. `bytesWritten` / `totalExpected` are mirrored on this side so the
/// Range header and progress reporting don't pay a FFI round-trip per read.
///
/// Carrying the context here is what lets the nonisolated URLSession delegate
/// methods run without any synchronous hop back to the main actor: everything
/// they read is either fixed at download-start time or lives on this writer.
///
/// Reference type so URLSession delegate callbacks can mutate fields without
/// re-storing into the table. All access is serialized on the owning
/// `WriterTable`'s queue, so it's safe across the main actor / delegate queue
/// boundary — hence `@unchecked Sendable`.
final class PartialWriter: @unchecked Sendable {
  let modelId: String
  /// The model this download belongs to. Snapshotted at start; immutable for
  /// the life of the transfer, so the delegate can report failures against it
  /// without looking the model up on the main actor.
  let model: Model
  /// HF cache root the finished blob is promoted into.
  let cacheDir: URL
  /// HF download plan (commit + repo dir) used by `writeBlobAndLink`.
  let plan: HFDownloadPlan
  let url: URL
  let filename: String
  let partialURL: URL
  /// Handle into the Rust core's streaming writer registry. Stable for the life
  /// of the transfer; `close()` releases it.
  let handle: UInt64
  /// Bytes currently on disk in the `.partial` file. Kept in step by `append`.
  var bytesWritten: Int64
  /// Full size of the remote file once known from Content-Range / Content-Length.
  /// 0 before the response arrives.
  var totalExpected: Int64
  /// SHA256 of the blob as advertised by HF (`X-Linked-Etag`), when available.
  /// Nil → we trust the computed digest instead.
  let expectedBlobHash: String?

  init(
    model: Model, cacheDir: URL, plan: HFDownloadPlan,
    url: URL, filename: String, partialURL: URL,
    handle: UInt64, bytesWritten: Int64, expectedBlobHash: String?
  ) {
    self.modelId = model.id
    self.model = model
    self.cacheDir = cacheDir
    self.plan = plan
    self.url = url
    self.filename = filename
    self.partialURL = partialURL
    self.handle = handle
    self.bytesWritten = bytesWritten
    self.totalExpected = 0
    self.expectedBlobHash = expectedBlobHash
  }

  /// Streams `data` to disk and into the core's running hash. Returns false when
  /// the write failed — the caller cancels the task and lets
  /// `didCompleteWithError` report it.
  func append(_ data: Data) -> Bool {
    guard LlamaCore.partialAppend(handle: handle, data: data) == 0 else {
      return false
    }
    bytesWritten += Int64(data.count)
    return true
  }

  /// The server answered 200 despite our Range request — restart the file from
  /// byte 0: the core truncates the `.partial`, resets the running hash, and
  /// resets the byte counter.
  @discardableResult
  func restart() -> Bool {
    guard LlamaCore.callValue("partial.restart", "ok", ["handle": Int64(handle)]) == true
    else { return false }
    bytesWritten = 0
    return true
  }

  /// Records the full remote size for progress reporting, mirrored to the core.
  func setTotalExpected(_ size: Int64) {
    totalExpected = size
    _ = LlamaCore.callJSON("partial.setTotalExpected", [
      "handle": Int64(handle),
      "size": size,
    ])
  }

  /// Digest over the bytes on disk so far, lowercase hex — ready at completion
  /// without a second full-file pass.
  func digest() -> String? {
    LlamaCore.callValue("partial.digest", "digest", ["handle": Int64(handle)])
  }

  /// Releases the core's writer (and its file handle) for this task.
  func close() {
    _ = LlamaCore.callJSON("partial.close", ["handle": Int64(handle)])
  }
}
