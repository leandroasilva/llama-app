import Foundation
import XCTest
@testable import Llama

final class HFCacheMigrationTests: XCTestCase {
  func testMergeCopiesBlobsAndPreservesSnapshotSymlinks() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("hf-cache-migration-\(UUID().uuidString)")
    let source = root.appendingPathComponent("source", isDirectory: true)
    let destination = root.appendingPathComponent("destination", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let blob = source.appendingPathComponent("blobs/abc123")
    let snapshot = source.appendingPathComponent("snapshots/revision/model.gguf")
    try FileManager.default.createDirectory(
      at: blob.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(
      at: snapshot.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("model".utf8).write(to: blob)
    try FileManager.default.createSymbolicLink(
      atPath: snapshot.path, withDestinationPath: "../../blobs/abc123")

    try HFCache.mergeDirectory(from: source, into: destination)

    let copiedBlob = destination.appendingPathComponent("blobs/abc123")
    let copiedSnapshot = destination.appendingPathComponent("snapshots/revision/model.gguf")
    XCTAssertEqual(try Data(contentsOf: copiedBlob), Data("model".utf8))
    XCTAssertEqual(
      try FileManager.default.destinationOfSymbolicLink(atPath: copiedSnapshot.path),
      "../../blobs/abc123")
    XCTAssertEqual(try Data(contentsOf: copiedSnapshot), Data("model".utf8))
  }

  func testMergeRejectsConflictingFiles() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("hf-cache-conflict-\(UUID().uuidString)")
    let source = root.appendingPathComponent("source", isDirectory: true)
    let destination = root.appendingPathComponent("destination", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let sourceBlob = source.appendingPathComponent("blobs/abc123")
    let destinationBlob = destination.appendingPathComponent("blobs/abc123")
    try FileManager.default.createDirectory(
      at: sourceBlob.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(
      at: destinationBlob.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("source".utf8).write(to: sourceBlob)
    try Data("other".utf8).write(to: destinationBlob)

    XCTAssertThrowsError(try HFCache.mergeDirectory(from: source, into: destination))
  }
}
