import XCTest

@testable import Llama

final class DownloadQueueTests: XCTestCase {
  func testPersistedQueueRoundTripsFIFOOrderAndModelData() throws {
    let first = QueuedModelDownload(
      model: Model(id: "org/first-GGUF:Q4_K_M", fileSize: 100, downloadUrl: URL(string: "https://huggingface.co/org/first/resolve/main/model.gguf")!),
      queuedAt: Date(timeIntervalSince1970: 1))
    let second = QueuedModelDownload(
      model: Model(id: "org/second-GGUF:Q4_K_M", fileSize: 200, downloadUrl: URL(string: "https://huggingface.co/org/second/resolve/main/model.gguf")!),
      queuedAt: Date(timeIntervalSince1970: 2))

    let encoded = try JSONEncoder().encode([first, second])
    let restored = try JSONDecoder().decode([QueuedModelDownload].self, from: encoded)

    XCTAssertEqual(restored.map { $0.model.id }, [first.model.id, second.model.id])
    XCTAssertEqual(restored.map(\.queuedAt), [first.queuedAt, second.queuedAt])
    XCTAssertEqual(restored.map { $0.model.fileSize }, [100, 200])
  }
}
