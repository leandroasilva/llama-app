import Foundation

/// Capabilities a UI or client setup flow can rely on without knowing which
/// inference binary provides them.
struct InferenceEngineCapabilities: OptionSet {
  let rawValue: UInt

  static let openAICompatibleAPI = Self(rawValue: 1 << 0)
  static let routerMode = Self(rawValue: 1 << 1)
  static let visionModels = Self(rawValue: 1 << 2)
  static let speculativeDecoding = Self(rawValue: 1 << 3)
}

struct InferenceEngineDescriptor: Equatable {
  let id: String
  let name: String
  let capabilities: InferenceEngineCapabilities
}

enum InferenceEngineState: Equatable {
  case stopped
  case starting
  case ready
  case failed(String)
}

/// Runtime contract for engines controlled by the app. Engine-specific model
/// formats and installers stay in adapters; the menu, agent setup and future
/// engine picker consume this small common surface.
@MainActor
protocol InferenceEngineRuntime: AnyObject {
  var descriptor: InferenceEngineDescriptor { get }
  var runtimeState: InferenceEngineState { get }
  var apiBaseURL: URL? { get }
  var activeModelID: String? { get }
  func startRuntime()
  func stopRuntime()
  func loadModel(id: String)
  func unloadModel(id: String)
}

/// Separates engine installation/update policy from runtime control.
@MainActor
protocol InferenceEngineInstaller: AnyObject {
  var installedVersion: String? { get }
  func installEngine() async -> Bool
}

/// Registry boundary: llama.cpp is the current engine, while new adapters can
/// be registered here without moving engine policy into the menu controller.
@MainActor
enum InferenceEngineRegistry {
  private struct Registration {
    let runtime: any InferenceEngineRuntime
    let installer: any InferenceEngineInstaller
  }

  private static var registrations: [String: Registration] = [
    "llama.cpp": Registration(runtime: LlamaServer.shared, installer: LlamaInstallManager.shared)
  ]

  static var current: any InferenceEngineRuntime { registrations["llama.cpp"]!.runtime }
  static var installer: any InferenceEngineInstaller { registrations["llama.cpp"]!.installer }
  static var available: [any InferenceEngineRuntime] {
    registrations.values.map(\.runtime).sorted { $0.descriptor.name < $1.descriptor.name }
  }

  static func runtime(for id: String) -> (any InferenceEngineRuntime)? {
    registrations[id]?.runtime
  }

  static func register(
    runtime: any InferenceEngineRuntime, installer: any InferenceEngineInstaller
  ) {
    registrations[runtime.descriptor.id] = Registration(runtime: runtime, installer: installer)
  }
}

extension LlamaServer: InferenceEngineRuntime {
  var descriptor: InferenceEngineDescriptor {
    InferenceEngineDescriptor(
      id: "llama.cpp",
      name: "llama.cpp",
      capabilities: [.openAICompatibleAPI, .routerMode, .visionModels, .speculativeDecoding])
  }

  var runtimeState: InferenceEngineState {
    switch state {
    case .idle: .stopped
    case .loading: .starting
    case .running: .ready
    case .error(let error): .failed(error.errorDescription ?? "Engine failed")
    }
  }

  var apiBaseURL: URL? {
    URL(string: "http://\(Self.localHost):\(Self.port)/v1")
  }

  var activeModelID: String? { activeModelId }

  func startRuntime() { LlamaInstallManager.shared.startServerWhenReady() }
  func stopRuntime() { stop() }

  func loadModel(id: String) {
    guard let model = ModelManager.shared.managedModels.first(where: { $0.id == id }) else { return }
    loadModel(model)
  }

  func unloadModel(id: String) {
    guard let model = ModelManager.shared.managedModels.first(where: { $0.id == id }) else { return }
    unloadModel(model)
  }
}

extension LlamaInstallManager: InferenceEngineInstaller {
  var installedVersion: String? { currentVersion?.tag }
  func installEngine() async -> Bool { await install() }
}
