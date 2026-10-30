import XCTest

@testable import Llama

final class ServerRuntimeMetricsTests: XCTestCase {
  func testParsesLlamaCppPrometheusMetrics() throws {
    let metrics = try XCTUnwrap(
      ServerRuntimeMetrics(prometheus: """
        # HELP llamacpp:predicted_tokens_seconds Average generation throughput
        llamacpp:predicted_tokens_seconds 47.25
        llamacpp:prompt_tokens_seconds 182.5
        llamacpp:requests_processing 1
        """))

    XCTAssertEqual(metrics.generationTokensPerSecond, 47.25)
    XCTAssertEqual(metrics.promptTokensPerSecond, 182.5)
    XCTAssertEqual(metrics.requestsProcessing, 1)
  }

  func testRejectsIncompleteOrNonFiniteMetrics() {
    XCTAssertNil(ServerRuntimeMetrics(prometheus: "llamacpp:requests_processing 0"))
    XCTAssertNil(
      ServerRuntimeMetrics(prometheus: """
        llamacpp:predicted_tokens_seconds NaN
        llamacpp:prompt_tokens_seconds 12
        llamacpp:requests_processing 1
        """))
  }
}
