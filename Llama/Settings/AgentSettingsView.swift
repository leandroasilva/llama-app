import SwiftUI

/// Copy-ready connection profiles for coding agents that speak the local
/// OpenAI-compatible API. Configuration is generated from the selected engine
/// and installed model, so customized ports and model IDs remain accurate.
struct AgentSettingsView: View {
  @State private var selectedModelID = ModelManager.shared.downloadedModels.first?.id ?? ""
  @State private var modelOptions = ModelManager.shared.downloadedModels
  @State private var copiedAgent: String?

  private var models: [Model] { modelOptions }
  private var selectedModel: Model? {
    models.first(where: { $0.id == selectedModelID }) ?? models.first
  }
  private var endpoint: URL? { InferenceEngineRegistry.current.apiBaseURL }
  private var usesAPIKey: Bool { LlamaServer.requestAPIKey != nil }

  var body: some View {
    Form {
      Section {
        SettingRow(
          title: "Model",
          description: "Choose the installed model the coding agent should use."
        ) {
          Picker("Model", selection: $selectedModelID) {
            ForEach(models, id: \.id) { model in
              Text(model.displayName).tag(model.id)
            }
          }
          .labelsHidden()
          .frame(maxWidth: 250)
          .disabled(models.isEmpty)
        }

        SettingRow(
          title: "API endpoint",
          description: "OpenAI-compatible API endpoint used by the copied agent settings."
        ) {
          Text(endpoint?.absoluteString ?? "llama.cpp is starting")
            .font(.system(.callout, design: .monospaced))
            .textSelection(.enabled)
        }
      }

      Section {
        VStack(alignment: .leading, spacing: 10) {
          Text("OpenCode")
            .font(.headline)
          Text("Copy a complete configuration for OpenCode's user-level or project-level opencode.json.")
            .font(.callout)
            .foregroundStyle(.secondary)
          copyButton(
            title: "Copy OpenCode configuration",
            agent: "OpenCode",
            content: openCodeConfiguration)
        }
        .padding(.vertical, 4)
      }

      Section {
        VStack(alignment: .leading, spacing: 10) {
          Text("Codex CLI")
            .font(.headline)
          Text("Codex uses llama.cpp's /v1/responses endpoint. The bundled engine supports it.")
            .font(.callout)
            .foregroundStyle(.secondary)
          copyButton(
            title: "Copy Codex config.toml",
            agent: "Codex",
            content: codexConfiguration)
        }
        .padding(.vertical, 4)
      }

      if usesAPIKey {
        Section {
          Text(
            "Network access is enabled. Add the API key from Settings > Network to the agent's environment as LLAMA_API_KEY; the copied configuration never contains the secret."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
        }
      }
    }
    .formStyle(.grouped)
    .onAppear(perform: refreshModels)
    .onReceive(NotificationCenter.default.publisher(for: .LBModelDownloadedListDidChange)) { _ in
      refreshModels()
    }
    .onChange(of: models.map(\.id)) { _, ids in
      if !ids.contains(selectedModelID), let first = ids.first {
        selectedModelID = first
      }
    }
  }

  private func refreshModels() {
    modelOptions = ModelManager.shared.downloadedModels
    if !modelOptions.contains(where: { $0.id == selectedModelID }),
      let first = modelOptions.first
    {
      selectedModelID = first.id
    }
  }

  private var openCodeConfiguration: String {
    guard let model = selectedModel, let endpoint else { return "" }
    var options: [String: Any] = ["baseURL": endpoint.absoluteString]
    if usesAPIKey { options["apiKey"] = "{env:LLAMA_API_KEY}" }
    let provider: [String: Any] = [
      "npm": "@ai-sdk/openai-compatible",
      "name": "Llama local",
      "options": options,
      "models": [model.id: ["name": model.displayName]],
    ]
    let config: [String: Any] = [
      "$schema": "https://opencode.ai/config.json",
      "provider": ["llama-local": provider],
      "model": "llama-local/\(model.id)",
    ]
    return prettyJSON(config)
  }

  private var codexConfiguration: String {
    guard let model = selectedModel, let endpoint else { return "" }
    let escapedModel = model.id.replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
    var lines = [
      "model = \"\(escapedModel)\"",
      "model_provider = \"llama_local\"",
      "",
      "[model_providers.llama_local]",
      "name = \"Llama local\"",
      "base_url = \"\(endpoint.absoluteString)\"",
      "wire_api = \"responses\"",
      "stream_idle_timeout_ms = 10000000",
    ]
    if usesAPIKey { lines.append("env_key = \"LLAMA_API_KEY\"") }
    return lines.joined(separator: "\n") + "\n"
  }

  private func prettyJSON(_ object: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(
      withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
      let json = String(data: data, encoding: .utf8)
    else { return "" }
    return json + "\n"
  }

  @ViewBuilder
  private func copyButton(title: String, agent: String, content: String) -> some View {
    HStack(spacing: 10) {
      Button(copiedAgent == agent ? "Copied" : title) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(content, forType: .string)
        copiedAgent = agent
      }
      .disabled(content.isEmpty)

      if copiedAgent == agent {
        Image(systemName: "checkmark.circle.fill")
          .foregroundStyle(.green)
          .accessibilityLabel("Copied")
      }
    }
  }
}
