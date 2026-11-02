import Foundation

enum CLIConfiguration {
  static func workspace(_ path: String?) throws -> URL {
    let raw = path ?? FileManager.default.currentDirectoryPath
    let expanded = NSString(string: raw).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw CLIError.usage("Project folder not found: \(raw)")
    }
    return url
  }

  static func baseURL(options: CLIOptions, environment: [String: String]) throws -> URL {
    let raw: String
    if let port = options.port {
      raw = "http://127.0.0.1:\(port)"
    } else if let attach = options.attach, !attach.isEmpty {
      raw = attach
    } else {
      raw = environment["LLAMA_AGENT_URL"] ?? "http://127.0.0.1:9931"
    }

    var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty else {
      throw CLIError.usage("The server URL is empty.")
    }
    if !value.contains("://") {
      value = "http://\(value)"
    }
    guard var components = URLComponents(string: value), components.host != nil else {
      throw CLIError.usage("Invalid server URL: \(raw)")
    }

    var path = components.path
    while path.hasSuffix("/") { path.removeLast() }
    if path.isEmpty {
      path = "/v1"
    } else if !path.hasSuffix("/v1") {
      path += "/v1"
    }
    components.path = path
    guard let url = components.url else {
      throw CLIError.usage("Invalid server URL: \(raw)")
    }
    return url
  }

  static func model(
    options: CLIOptions, server: ServerClient, environment: [String: String]
  ) async throws -> String {
    if let model = options.model?.trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty {
      return model
    }
    if let model = environment["LLAMA_AGENT_MODEL"]?
      .trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty
    {
      return model
    }
    let models = try await server.models()
    guard let first = models.first else { throw CLIError.noModels }
    return first
  }

  static func contextTokens(options: CLIOptions, environment: [String: String]) -> Int {
    if let tokens = options.contextTokens { return tokens }
    if let raw = environment["LLAMA_AGENT_CTX"], let tokens = Int(raw), tokens >= 512 {
      return tokens
    }
    return 8_192
  }

  static func apiKey(environment: [String: String]) -> String? {
    guard let key = environment["LLAMA_API_KEY"], !key.isEmpty else { return nil }
    return key
  }
}
