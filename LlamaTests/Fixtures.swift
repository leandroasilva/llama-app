import Foundation

/// Locates the shared `Fixtures/` corpus the Swift and Rust test suites both
/// consume.
///
/// The folder is copied into the test bundle (a folder reference in the
/// LlamaTests target), so tests read it from the bundle at runtime. That keeps
/// them working no matter where the checkout lives — in particular it avoids
/// macOS's removable-volume consent prompt, which blocks `open()` from the
/// signed test host when the repo sits on an external disk.
enum Fixtures {
  /// The `Fixtures` directory: the copy in the test bundle when present,
  /// otherwise the source tree (`#filePath`) for running from a non-bundled
  /// context.
  static let directory: URL = {
    if let bundled = bundleFixtures, FileManager.default.fileExists(atPath: bundled.path) {
      return bundled
    }
    return URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()  // LlamaTests/
      .deletingLastPathComponent()  // repo root
      .appendingPathComponent("Fixtures")
  }()

  private static var bundleFixtures: URL? {
    Bundle(for: FixtureBundleToken.self)
      .resourceURL?
      .appendingPathComponent("Fixtures")
  }
}

/// Anchors `Bundle(for:)` to the test bundle.
private final class FixtureBundleToken {}
