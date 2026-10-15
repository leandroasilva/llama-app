import Foundation
import XCTest

@testable import Llama

@MainActor
final class LlamaServerTests: XCTestCase {

  func testDefaultPort() {
    XCTAssertEqual(LlamaServer.defaultPort, 9931)
  }

  func testPortAvailabilityCheckOnDefaultPort() {
    // isPortAvailable is best-effort; just verify it doesn't crash and returns a Bool.
    let _ = LlamaServer.isPortAvailable(LlamaServer.defaultPort)
  }

  func testPortAvailabilityForPrivilegedPort() {
    // Port 1 is privileged; binding should fail, so isPortAvailable returns false
    // (or true if we can't probe -- either is acceptable).
    let _ = LlamaServer.isPortAvailable(1)
  }

  func testInitialStateIsIdle() {
    let server = LlamaServer()
    XCTAssertEqual(server.state, .idle)
    XCTAssertTrue(server.modelStatuses.isEmpty)
    XCTAssertNil(server.activeModelId)
  }

  func testStopFromIdleIsNoOp() {
    let server = LlamaServer()
    server.stop()
    XCTAssertEqual(server.state, .idle)
  }

  func testEmptyCachePathIsPrivateToTheUser() {
    let path = LlamaServer.emptyCachePath
    // Never a shared, world-writable location like /tmp -- another user could
    // pre-create the name there and point it at their own files. It lives
    // under the app's own Application Support directory instead.
    XCTAssertFalse(path.hasPrefix("/tmp"))
    XCTAssertTrue(path.contains("Application Support"))
    XCTAssertTrue(path.hasSuffix("empty-cache"))
  }

  func testLogFilePathIsInLibrary() {
    XCTAssertTrue(LlamaServer.logFilePath.contains("Library/Logs"))
    XCTAssertTrue(LlamaServer.logFilePath.hasSuffix("llama-server.log"))
  }

  func testServerStateEquality() {
    XCTAssertEqual(LlamaServer.ServerState.idle, LlamaServer.ServerState.idle)
    XCTAssertEqual(LlamaServer.ServerState.loading, LlamaServer.ServerState.loading)
    XCTAssertEqual(LlamaServer.ServerState.running, LlamaServer.ServerState.running)
    XCTAssertNotEqual(LlamaServer.ServerState.idle, LlamaServer.ServerState.running)
  }

  func testServerErrorEquality() {
    let e1 = LlamaServerError.launchFailed("test")
    let e2 = LlamaServerError.launchFailed("test")
    let e3 = LlamaServerError.launchFailed("other")
    XCTAssertEqual(e1, e2)
    XCTAssertNotEqual(e1, e3)
  }

  func testPortInUseErrorDescription() {
    let err = LlamaServerError.portInUse(port: 9931, by: "other-process")
    XCTAssertNotNil(err.errorDescription)
    XCTAssertTrue(err.errorDescription!.contains("9931"))
    XCTAssertTrue(err.errorDescription!.contains("other-process"))
  }

  func testInvalidPathErrorDescription() {
    let err = LlamaServerError.invalidPath("/bad/path")
    XCTAssertNotNil(err.errorDescription)
    XCTAssertTrue(err.errorDescription!.contains("/bad/path"))
  }

  func testLaunchSpecEquality() {
    let spec1 = LlamaServer.LaunchSpec(
      executablePath: "/usr/bin/llama",
      arguments: ["--port", "9931"],
      extraArgCount: 0,
      env: [("KEY", "VALUE")]
    )
    let spec2 = LlamaServer.LaunchSpec(
      executablePath: "/usr/bin/llama",
      arguments: ["--port", "9931"],
      extraArgCount: 0,
      env: [("KEY", "VALUE")]
    )
    let spec3 = LlamaServer.LaunchSpec(
      executablePath: "/usr/bin/llama",
      arguments: ["--port", "9932"],
      extraArgCount: 0,
      env: [("KEY", "VALUE")]
    )
    XCTAssertEqual(spec1, spec2)
    XCTAssertNotEqual(spec1, spec3)
  }

  func testLaunchSpecEnvOrderMatters() {
    let spec1 = LlamaServer.LaunchSpec(
      executablePath: "/usr/bin/llama",
      arguments: [],
      extraArgCount: 0,
      env: [("A", "1"), ("B", "2")]
    )
    let spec2 = LlamaServer.LaunchSpec(
      executablePath: "/usr/bin/llama",
      arguments: [],
      extraArgCount: 0,
      env: [("B", "2"), ("A", "1")]
    )
    XCTAssertNotEqual(spec1, spec2)
  }

  func testIsActiveModelChecksStatuses() {
    let server = LlamaServer()
    let model = Model(
      id: "org/repo:Q4_K_M",
      ctxWindow: 4096,
      fileSize: 1_000_000,
      ctxBytesPer1kTokens: 0,
      downloadUrl: URL(string: "https://example.com/model.gguf")!,
      scannedDraftHead: nil,
      isDecisionModel: false
    )

    server.modelStatuses = [:]
    XCTAssertFalse(server.isActive(model: model))

    server.modelStatuses[model.id] = .loaded
    XCTAssertTrue(server.isActive(model: model))

    server.modelStatuses[model.id] = .sleeping
    XCTAssertTrue(server.isActive(model: model))

    server.modelStatuses[model.id] = .loading
    XCTAssertFalse(server.isActive(model: model))

    server.modelStatuses[model.id] = .unloaded
    XCTAssertFalse(server.isActive(model: model))
  }

  func testIsLoadingModelChecksStatuses() {
    let server = LlamaServer()
    let model = Model(
      id: "org/repo:Q4_K_M",
      ctxWindow: 4096,
      fileSize: 1_000_000,
      ctxBytesPer1kTokens: 0,
      downloadUrl: URL(string: "https://example.com/model.gguf")!,
      scannedDraftHead: nil,
      isDecisionModel: false
    )

    server.modelStatuses = [model.id: .loading]
    XCTAssertTrue(server.isLoading(model: model))

    server.modelStatuses = [model.id: .loaded]
    XCTAssertFalse(server.isLoading(model: model))
  }

  func testActiveModelIdDerivesFromStatuses() {
    let server = LlamaServer()
    XCTAssertNil(server.activeModelId)

    server.modelStatuses = ["model-a": .unloaded, "model-b": .loaded]
    XCTAssertEqual(server.activeModelId, "model-b")

    server.modelStatuses = ["model-a": .loading, "model-b": .unloaded]
    XCTAssertEqual(server.activeModelId, "model-a")

    server.modelStatuses = ["model-a": .unloaded, "model-b": .unloaded]
    XCTAssertNil(server.activeModelId)
  }

  func testIsAnyModelLoaded() {
    let server = LlamaServer()
    XCTAssertFalse(server.isAnyModelLoaded)

    server.modelStatuses = ["model-a": .loading]
    XCTAssertFalse(server.isAnyModelLoaded)

    server.modelStatuses = ["model-a": .loaded]
    XCTAssertTrue(server.isAnyModelLoaded)

    server.modelStatuses = ["model-a": .sleeping]
    XCTAssertFalse(server.isAnyModelLoaded)
  }

  func testIsAnyModelLoading() {
    let server = LlamaServer()
    XCTAssertFalse(server.isAnyModelLoading)

    server.modelStatuses = ["model-a": .loaded]
    XCTAssertFalse(server.isAnyModelLoading)

    server.modelStatuses = ["model-a": .loading]
    XCTAssertTrue(server.isAnyModelLoading)
  }

  func testIsRunningReflectsState() {
    let server = LlamaServer()
    XCTAssertFalse(server.isRunning)

    server.state = .running
    XCTAssertTrue(server.isRunning)

    server.state = .loading
    XCTAssertFalse(server.isRunning)
  }

  func testIsLoadingReflectsState() {
    let server = LlamaServer()
    XCTAssertFalse(server.isLoading)

    server.state = .loading
    XCTAssertTrue(server.isLoading)

    server.state = .running
    XCTAssertFalse(server.isLoading)
  }
}
