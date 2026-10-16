import Foundation

/// Describes the volume currently hosting the model cache. This intentionally
/// reports location, not an invented throughput score: external SSDs can be
/// fast and some internal paths can still be slow under pressure.
enum StorageHealth {
  enum Location: Equatable {
    case unavailable
    case internalVolume
    case externalVolume
  }

  static func location(of directory: URL) -> Location {
    guard let probe = nearestExistingURL(to: directory) else { return .unavailable }

    if let values = try? probe.resourceValues(forKeys: [.volumeIsInternalKey]),
      let isInternal = values.volumeIsInternal
    {
      return isInternal ? .internalVolume : .externalVolume
    }

    // Fallback for filesystems that don't report volumeIsInternal. macOS mounts
    // ordinary external volumes below /Volumes, while the default cache lives
    // below the user's home directory.
    return directory.path == "/Volumes" || directory.path.hasPrefix("/Volumes/")
      ? .externalVolume
      : .internalVolume
  }

  static func volumeName(of directory: URL) -> String? {
    guard let probe = nearestExistingURL(to: directory) else { return nil }
    return try? probe.resourceValues(forKeys: [.volumeNameKey]).volumeName
  }

  private static func nearestExistingURL(to directory: URL) -> URL? {
    var candidate = directory
    let fileManager = FileManager.default
    while !fileManager.fileExists(atPath: candidate.path) {
      let parent = candidate.deletingLastPathComponent()
      guard parent.path != candidate.path else { return nil }
      candidate = parent
    }
    return candidate
  }
}
