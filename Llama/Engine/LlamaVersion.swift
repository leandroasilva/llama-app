import Foundation

/// A llama.cpp build version. Comparison is by build number -- the monotonic
/// `bXXXX` counter llama.cpp bumps each release; commit shas and semantic
/// versions are ignored.
///
/// The parse lives in the Rust core (`version.parse`); this type is a thin
/// view so call sites keep working unchanged.
struct LlamaVersion: Comparable, CustomStringConvertible {
  /// The build number, e.g. `9370` for `b9370`.
  let build: Int

  /// Parses a pinned tag (`b9370`) or `llama version` output, which comes in
  /// two formats:
  /// - `b9370-aa50b2c2a` -- builds before b10398
  /// - `version: 0.5.0-dev (build 11200, commit 81bc6b83f)` -- b10398 on, when
  ///   llama.cpp introduced semantic versioning; the build number moved into
  ///   the parenthetical, so the leading token is no longer the version
  /// Returns nil if no build number can be read.
  init?(parsing output: String) {
    guard let build: Int = LlamaCore.callValue("version.parse", "build", ["output": output])
    else { return nil }
    self.build = build
  }

  /// The build tag, e.g. `b9370`. For display.
  var tag: String { "b\(build)" }

  static func < (lhs: LlamaVersion, rhs: LlamaVersion) -> Bool { lhs.build < rhs.build }

  var description: String { tag }
}
