import Foundation

struct ServerClient {
  let baseURL: URL
  let apiKey: String?
  let session: URLSession

  init(baseURL: URL, apiKey: String?, session: URLSession = ServerClient.defaultSession) {
    self.baseURL = baseURL
    self.apiKey = apiKey
    self.session = session
  }

  static let defaultSession: URLSession = {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.urlCache = nil
    configuration.timeoutIntervalForRequest = 30
    configuration.timeoutIntervalForResource = 60
    return URLSession(configuration: configuration)
  }()

  func models() async throws -> [String] {
    var request = URLRequest(url: baseURL.appendingPathComponent("models"))
    request.httpMethod = "GET"
    request.timeoutInterval = 15
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (data, response): (Data, URLResponse)
    do {
      (data, response) = try await session.data(for: request)
    } catch {
      throw CLIError.server(
        "Couldn't reach the local server at \(baseURL.absoluteString): \(error.localizedDescription)")
    }

    guard let http = response as? HTTPURLResponse else {
      throw CLIError.server("The local server sent a response llama-agent couldn't read.")
    }
    guard (200..<300).contains(http.statusCode) else {
      let body = String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      throw CLIError.server(
        body.isEmpty
          ? "The local server returned HTTP \(http.statusCode)."
          : "The local server returned HTTP \(http.statusCode): \(body)")
    }

    struct Response: Decodable { let data: [Model] }
    struct Model: Decodable { let id: String }

    do {
      return try JSONDecoder().decode(Response.self, from: data).data.map(\.id)
    } catch {
      throw CLIError.server("The local server's model list couldn't be read.")
    }
  }
}
