import XCTest

@testable import Llama

final class InferenceEngineRuntimeTests: XCTestCase {
  @MainActor
  func testLlamaCppIsRegisteredThroughTheCommonRuntimeContract() {
    let engine = InferenceEngineRegistry.current

    XCTAssertEqual(engine.descriptor.id, "llama.cpp")
    XCTAssertTrue(engine.descriptor.capabilities.contains(.openAICompatibleAPI))
    XCTAssertTrue(engine.descriptor.capabilities.contains(.routerMode))
    XCTAssertEqual(engine.apiBaseURL?.path, "/v1")
    XCTAssertTrue(InferenceEngineRegistry.available.contains { $0.descriptor.id == engine.descriptor.id })
  }
}
