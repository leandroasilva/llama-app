import XCTest

@testable import Llama

final class PerformanceConfigTests: XCTestCase {
  func testDefaultsFavorLargerContextWithLowerMemoryUse() {
    let config = PerformanceConfig.default

    XCTAssertTrue(config.flashAttention)
    XCTAssertEqual(config.kvCacheTypeK, .q8_0)
    XCTAssertEqual(config.kvCacheTypeV, .q8_0)
    XCTAssertEqual(config.parallel, 1)
    XCTAssertEqual(config.batchSize, 512)
    XCTAssertEqual(config.ubatchSize, 256)
    XCTAssertFalse(config.noKVOffload)
  }

  func testIniPairsUseLlamaPresetKeys() {
    let pairs = Dictionary(
      uniqueKeysWithValues: PerformanceConfig.default.iniPairs.map { ($0.key, $0.value) })

    XCTAssertEqual(pairs["flash-attn"], "on")
    XCTAssertEqual(pairs["cache-type-k"], "q8_0")
    XCTAssertEqual(pairs["cache-type-v"], "q8_0")
    XCTAssertEqual(pairs["parallel"], "1")
    XCTAssertEqual(pairs["batch-size"], "512")
    XCTAssertEqual(pairs["ubatch-size"], "256")
    XCTAssertEqual(pairs["no-kv-offload"], "false")
  }

  func testNormalizationClampsUnsafeValues() {
    var config = PerformanceConfig.default
    config.parallel = 0
    config.batchSize = 999
    config.ubatchSize = 2048

    let normalized = config.normalized

    XCTAssertEqual(normalized.parallel, 1)
    XCTAssertEqual(normalized.batchSize, 512)
    XCTAssertEqual(normalized.ubatchSize, 512)
  }

  func testNormalizationKeepsValidValues() {
    var config = PerformanceConfig.default
    config.parallel = 4
    config.batchSize = 2048
    config.ubatchSize = 1024

    XCTAssertEqual(config.normalized, config)
  }
}
