import SwiftUI

/// Installs the bundled command-line agent so it can be run from Terminal.
struct AgentSettingsView: View {
  @State private var isInstallingAgent = false
  @State private var agentInstallMessage: String?

  var body: some View {
    Form {
      Section {
        VStack(alignment: .leading, spacing: 10) {
          Text("Command-line agent")
            .font(.headline)
          Text(
            "Run the local coding agent in Terminal with the llama-agent command. "
              + "The installer links it to this app, so app updates keep it current."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)

          HStack(spacing: 10) {
            Button {
              installAgentCommand()
            } label: {
              Text(
                isInstallingAgent
                  ? "Installing…"
                  : (agentCommandInstalled ? "Reinstall command" : "Install command")
              )
            }
            .disabled(isInstallingAgent || LlamaAgentInstaller.bundledURL == nil)

            if agentCommandInstalled {
              Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
            }
          }

          if let agentInstallMessage {
            Text(agentInstallMessage)
              .font(.caption)
              .foregroundStyle(.secondary)
              .fixedSize(horizontal: false, vertical: true)
          } else if LlamaAgentInstaller.bundledURL == nil {
            Text("This build doesn't include the llama-agent binary.")
              .font(.caption)
              .foregroundStyle(.secondary)
          } else {
            Text("Installs at \(LlamaAgentInstaller.installPath).")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
        .padding(.vertical, 4)
      }
    }
    .formStyle(.grouped)
  }

  private var agentCommandInstalled: Bool {
    LlamaAgentInstaller.isInstalled && LlamaAgentInstaller.isLinkedToBundle
  }

  private func installAgentCommand() {
    isInstallingAgent = true
    agentInstallMessage = nil
    Task { @MainActor in
      let result = await LlamaAgentInstaller.install()
      isInstallingAgent = false
      switch result {
      case .success:
        agentInstallMessage = "Installed. Open a new terminal and run: llama-agent --help"
      case .failure(let error):
        agentInstallMessage = error.localizedDescription
      }
    }
  }
}
