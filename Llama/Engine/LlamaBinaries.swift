import Foundation
import os.log

/// Resolves the `llama` executable the app drives, and classifies whether the
/// app may update it.
///
/// The app follows a shared-path model:
/// - it manages the curl-install path (`~/.llama-app/llama`, what `install.sh`
///   produces): it may install a binary there and keep it updated
/// - any other install (e.g. Homebrew) is left unmanaged: the app uses it
///   but never modifies it
///
/// `llama` is the unified llama.cpp executable -- the server is `llama serve`
/// and memory profiling is `llama fit-params`, both subcommands of this one
/// binary. There is no separate `llama-server` / `llama-fit-params` to find.
enum LlamaBinaries {

  private static let logger = Logger(subsystem: Logging.subsystem, category: "LlamaBinaries")

  /// The curl-install path the app manages (matches `install.sh`'s layout).
  /// The real binary lives in `~/.llama-app`; `install.sh` also drops a
  /// `~/.local/bin/llama` symlink onto PATH, but the app points at the real file.
  /// Computed by the core (`engine.managedPath`).
  static let managedPath: String =
    LlamaCore.callValue("engine.managedPath", "path", ["home": NSHomeDirectory()])
    ?? (NSHomeDirectory() as NSString).appendingPathComponent(".llama-app/llama")

  /// The build the app installs and keeps its own binary at -- the pinned
  /// target, not whatever is newest. Bump per app release after smoke-testing
  /// `serve` + `fit-params`; the app's auto-updater then rolls it out.
  static let targetVersion = LlamaVersion(parsing: "b11429")!

  /// The minimum build the app accepts from an unmanaged install (e.g. Homebrew)
  /// before nudging the user to update -- the app can't update those itself.
  /// Must be <= targetVersion. Set to the build that introduced DFlash
  /// drafting (`spec-type = draft-dflash`, llama.cpp PR #22105), the newest
  /// thing the app relies on -- older builds reject the spec type, so every
  /// model with a DFlash head would fail to load. The previous floor, b9726,
  /// was the `--agent` serve flag (llama.cpp PR #24801), which this build
  /// includes. Only raise this if the app starts relying on an even newer
  /// flag or value.
  static let floorVersion = LlamaVersion(parsing: "b9831")!

  /// Where the in-use binary comes from. Only `managed` is the app's to
  /// update; `brew` and `external` are both used as-is and never modified.
  /// Brew is split out so the footer can hint at the actual update channel
  /// (`brew upgrade`) instead of a generic "external" marker.
  enum Origin: Equatable {
    case managed, brew, external

    init?(_ raw: String) {
      switch raw {
      case "managed": self = .managed
      case "brew": self = .brew
      case "external": self = .external
      default: return nil
      }
    }
  }

  /// Resolves the active `llama` binary: the managed path wins, then the
  /// unmanaged locations in order. Nil when none is found, which is what sends
  /// the app down the install flow. The resolution (including Homebrew's Cellar
  /// detection) lives in the core (`engine.binariesResolve`).
  static func resolve() -> (path: String, origin: Origin)? {
    var ignore = false
    #if DEBUG
      // Dev affordance: pretend unmanaged installs (e.g. Homebrew) aren't present,
      // so the missing -> install path can be exercised on a machine that already
      // has llama.cpp. Toggle with:
      //   defaults write app.llama.Llama.dev ignoreUnmanagedLlama -bool YES
      if UserDefaults.standard.bool(forKey: "ignoreUnmanagedLlama") {
        logger.debug("ignoreUnmanagedLlama set; ignoring unmanaged installs")
        ignore = true
      }
    #endif

    guard let out = LlamaCore.callJSON("engine.binariesResolve", [
      "home": NSHomeDirectory(),
      "ignoreUnmanaged": ignore,
    ]), let path = out["path"] as? String, let raw = out["origin"] as? String,
      let origin = Origin(raw)
    else { return nil }
    return (path, origin)
  }

  /// The path to the `llama` binary to invoke, or `nil` if none is installed.
  static var llamaPath: String? {
    guard let path = resolve()?.path else {
      logger.error("No llama binary found")
      return nil
    }
    logger.debug("Using llama binary at \(path, privacy: .public)")
    return path
  }

  /// Reads the version reported by the binary at `path`, or nil if it can't be
  /// run or its output can't be parsed. The subprocess and parse live in the
  /// core (`engine.readVersion`); it just prints the build and exits -- no model
  /// load. Blocks on the subprocess, so call off the main thread.
  static func readVersion(at path: String) -> LlamaVersion? {
    guard let build: Int = LlamaCore.callValue("engine.readVersion", "build", ["path": path])
    else { return nil }
    return LlamaVersion(build: build)
  }
}
