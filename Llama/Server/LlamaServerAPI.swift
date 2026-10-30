import Foundation
import os.log

/// Per-model load state as reported by llama-server's `/models` endpoint.
enum ModelLoadState: String, Equatable {
  case loaded
  case loading
  case sleeping
  case unloaded
}

/// Live throughput and request activity exported by llama.cpp's Prometheus
/// `/metrics` endpoint. These gauges are aggregate engine metrics; they never
/// include prompt or completion text.
struct ServerRuntimeMetrics: Equatable {
  let promptTokensPerSecond: Double
  let generationTokensPerSecond: Double
  let requestsProcessing: Int

  init?(prometheus: String) {
    var values: [String: Double] = [:]
    for line in prometheus.split(whereSeparator: \.isNewline) {
      guard !line.hasPrefix("#"), let separator = line.lastIndex(of: " ") else { continue }
      let rawName = line[..<separator]
      let name = rawName.prefix { $0 != "{" }.split(separator: ":").last.map(String.init) ?? ""
      guard let value = Double(line[line.index(after: separator)...]), value.isFinite else { continue }
      values[name] = value
    }

    guard let prompt = values["prompt_tokens_seconds"],
      let generation = values["predicted_tokens_seconds"],
      let requests = values["requests_processing"]
    else { return nil }

    promptTokensPerSecond = max(0, prompt)
    generationTokensPerSecond = max(0, generation)
    requestsProcessing = max(0, Int(requests))
  }
}

/// HTTP client for communicating with llama-server's REST API.
/// Encapsulates request building and response parsing for server endpoints.
struct LlamaServerAPI {
  private let logger = Logger(subsystem: Logging.subsystem, category: "LlamaServerAPI")

  // MARK: - Public API

  /// Requests the server to load a model by ID.
  /// Returns true if the request was sent successfully.
  func loadModel(id: String) async -> Bool {
    await post(endpoint: "models/load", body: ["model": id])
  }

  /// Requests the server to unload a model by ID.
  /// Returns true if the server acknowledged the request.
  func unloadModel(id: String) async -> Bool {
    await post(endpoint: "models/unload", body: ["model": id])
  }

  /// Asks the router server to re-read models.ini and reconcile its model list
  /// in place (`GET /models?reload=1`). The server only unloads models whose
  /// entry was removed or changed -- untouched models stay resident.
  /// Returns true if the server acknowledged the request.
  func reloadModels() async -> Bool {
    await get(endpoint: "models?reload=1") != nil
  }

  /// Fetches the current status of all models.
  /// Returns a dictionary mapping model IDs to their load state.
  /// Unknown or missing states are treated as `.unloaded`.
  func fetchModelStatuses() async -> [String: ModelLoadState]? {
    guard let data = await get(endpoint: "models") else { return nil }

    guard
      let statuses: [[String]] = LlamaCore.callValue(
        "server.parseModelStatuses", "statuses", ["body": String(decoding: data, as: UTF8.self)])
    else { return nil }

    return statuses.reduce(into: [String: ModelLoadState]()) { dict, pair in
      guard pair.count == 2 else { return }
      dict[pair[0]] = ModelLoadState(rawValue: pair[1]) ?? .unloaded
    }
  }

  /// Fetches Prometheus gauges for one model in llama-server's router.
  func fetchRuntimeMetrics(modelId: String) async -> ServerRuntimeMetrics? {
    guard var components = URLComponents(string: "\(baseUrl)/metrics") else { return nil }
    components.queryItems = [URLQueryItem(name: "model", value: modelId)]
    guard let url = components.url, let data = await get(url: url, timeout: 2.0) else { return nil }
    return ServerRuntimeMetrics(prometheus: String(decoding: data, as: UTF8.self))
  }

  // MARK: - Private Helpers

  // Always the current effective host and port -- read live so a runtime change
  // (port, or a new network bind address) is picked up without recreating the
  // client. The host matters: with a specific bind address the server isn't on
  // loopback at all, so localhost would silently fail every request.
  private var baseUrl: String { "http://\(LlamaServer.localHost):\(LlamaServer.port)" }

  /// A session that never caches.
  ///
  /// Every endpoint here either reports live server state or has a side effect
  /// (`?reload=1`), so a cached response is always wrong -- and a cached 200 is
  /// worse than an error, because the caller reads it as "the server did the
  /// thing". `llama serve` sends no cache headers at all, which leaves the
  /// decision to CFNetwork's heuristics; the shared cache has been seen holding
  /// entries for these URLs. Opting out is cheaper than relying on that
  /// heuristic staying favourable.
  private static let session: URLSession = {
    let config = URLSessionConfiguration.default
    config.urlCache = nil
    return URLSession(configuration: config)
  }()

  /// Authorization header for a server that was started with an API key. Empty
  /// on a loopback server (no key), so those requests go out header-free.
  private var authHeaders: [String: String] {
    guard let key = LlamaServer.requestAPIKey else { return [:] }
    return ["Authorization": "Bearer \(key)"]
  }

  /// Sends a GET request and returns the response data.
  private func get(endpoint: String, timeout: TimeInterval = 2.0) async -> Data? {
    guard let url = URL(string: "\(baseUrl)/\(endpoint)") else { return nil }

    return await get(url: url, timeout: timeout)
  }

  private func get(url: URL, timeout: TimeInterval) async -> Data? {

    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    for (field, value) in authHeaders {
      request.setValue(value, forHTTPHeaderField: field)
    }

    do {
      let (data, response) = try await Self.session.data(for: request)
      guard let httpResponse = response as? HTTPURLResponse,
        httpResponse.statusCode == 200
      else { return nil }
      return data
    } catch {
      return nil
    }
  }

  /// Sends a POST request with JSON body.
  /// Returns true if the request succeeded (2xx status).
  private func post(endpoint: String, body: [String: Any]) async -> Bool {
    guard let url = URL(string: "\(baseUrl)/\(endpoint)") else { return false }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    for (field, value) in authHeaders {
      request.setValue(value, forHTTPHeaderField: field)
    }
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)

    do {
      let (_, response) = try await Self.session.data(for: request)
      return (response as? HTTPURLResponse)?.statusCode == 200
    } catch {
      return false
    }
  }

}
