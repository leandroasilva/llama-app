import AppKit
import SwiftUI

/// Owns the built-in coding agent's window. The session lives as long as the
/// window: closing it cancels any in-flight generation, and reopening it keeps
/// the conversation so a half-finished task isn't lost by a stray click.
@MainActor
final class AgentWindowController: NSObject, NSWindowDelegate {
  static let shared = AgentWindowController()

  private var window: NSWindow?
  private(set) var session: AgentSession?

  var hasVisibleWindow: Bool { window?.isVisible == true }

  func show() {
    if window == nil {
      let stored = UserSettings.agentProjectPath.map { URL(fileURLWithPath: $0) }
      let session = AgentSession(workspaceURL: stored)
      self.session = session

      let host = NSHostingController(rootView: AgentView(session: session))
      let window = NSWindow(contentViewController: host)
      window.title = "Llama Agent"
      window.setContentSize(NSSize(width: 880, height: 680))
      window.minSize = NSSize(width: 700, height: 520)
      window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
      window.isReleasedWhenClosed = false
      window.delegate = self
      window.center()
      self.window = window
    }

    NSApp.setActivationPolicy(.regular)
    window?.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
  }

  func windowWillClose(_ notification: Notification) {
    session?.cancel()
    if !SettingsWindowController.shared.hasVisibleWindow {
      NSApp.setActivationPolicy(.accessory)
    }
  }
}

struct AgentView: View {
  let session: AgentSession

  @State private var draft = ""
  @State private var models = ModelManager.shared.downloadedModels
  @State private var contextNotice: ContextNotice?
  @FocusState private var composerFocused: Bool

  private let bottomID = "agent-bottom"
  private let horizontalPadding: CGFloat = 18

  var body: some View {
    VStack(spacing: 0) {
      toolbar
      if let notice = contextNotice {
        Divider()
        contextBanner(notice)
      }
      Divider()
      transcript
      Divider()
      if let approval = session.pendingApproval {
        AgentApprovalCard(approval: approval, session: session)
          .id(approval.id)
        Divider()
      }
      composer
    }
    .frame(minWidth: 700, minHeight: 520)
    .background(Color(nsColor: .windowBackgroundColor))
    .onAppear {
      refreshModels()
      composerFocused = true
    }
    .onReceive(NotificationCenter.default.publisher(for: .LBModelDownloadedListDidChange)) { _ in
      refreshModels()
    }
    .onChange(of: session.modelID) { _, _ in
      refreshContextNotice()
    }
  }

  // MARK: - Toolbar

  private var toolbar: some View {
    HStack(spacing: 12) {
      Picker(
        "Model",
        selection: Binding(
          get: { session.modelID },
          set: { session.setModel($0) })
      ) {
        if models.isEmpty {
          Text("No models installed").tag("")
        }
        ForEach(models) { model in
          Text(model.displayName).tag(model.id)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 280)
      .disabled(session.isRunning || models.isEmpty)

      Button {
        chooseProject()
      } label: {
        Label(projectLabel, systemImage: "folder")
          .lineLimit(1)
      }
      .disabled(session.isRunning)
      .help("Choose the folder the agent can read and change")

      Spacer()

      if session.isRunning {
        ProgressView()
          .controlSize(.small)
          .accessibilityLabel("Agent working")
      }

      Button {
        session.reset()
      } label: {
        Label("New", systemImage: "square.and.pencil")
      }
      .disabled(session.isRunning || session.entries.isEmpty)
      .help("Start a new conversation")
    }
    .padding(.horizontal, horizontalPadding)
    .padding(.vertical, 10)
  }

  private struct ContextNotice {
    let current: ContextTier
    let target: ContextTier?
    let pinned: Bool
  }

  private var currentModel: Model? {
    models.first { $0.id == session.modelID }
  }

  private var projectLabel: String {
    guard let url = session.workspaceURL else { return "Choose project…" }
    return abbreviatedPath(url, maxLength: 44)
  }

  // MARK: - Transcript

  private var transcript: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 12) {
          if session.entries.isEmpty {
            emptyState
          }
          ForEach(session.entries) { entry in
            AgentEntryView(entry: entry)
          }
          Color.clear.frame(height: 1).id(bottomID)
        }
        .padding(.horizontal, horizontalPadding)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
      }
      .onChange(of: session.scrollTick) { _, _ in
        proxy.scrollTo(bottomID, anchor: .bottom)
      }
    }
  }

  private var emptyState: some View {
    VStack(alignment: .leading, spacing: 12) {
      Image(systemName: "terminal")
        .font(.system(size: 26, weight: .light))
        .foregroundStyle(.secondary)

      Text("A coding agent on your Mac")
        .font(.title3.weight(.semibold))

      Text(
        "It reads and edits files, runs commands, and asks before changing anything. "
          + "Everything runs against the model installed in Llama; nothing leaves this Mac."
      )
      .foregroundStyle(.secondary)
      .fixedSize(horizontal: false, vertical: true)

      if session.workspaceURL == nil {
        Button("Choose project folder…") { chooseProject() }
          .controlSize(.large)
      } else if let url = session.workspaceURL {
        Label(abbreviatedPath(url, maxLength: 72), systemImage: "folder.fill")
          .font(.callout)
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }

      if models.isEmpty {
        Text("Install a model from the menu bar first. The agent needs one to run.")
          .font(.callout)
          .foregroundStyle(.orange)
      } else {
        Text("Works best with a model trained for tool calling, such as a coding model.")
          .font(.callout)
          .foregroundStyle(.secondary)
      }
    }
    .padding(.top, 48)
    .padding(.bottom, 24)
  }

  // MARK: - Composer

  private var composer: some View {
    HStack(alignment: .bottom, spacing: 10) {
      ZStack(alignment: .topLeading) {
        TextEditor(text: $draft)
          .font(.body)
          .scrollContentBackground(.hidden)
          .focused($composerFocused)
          .onKeyPress(.return, phases: .down) { press in
            guard press.modifiers.contains(.command) else { return .ignored }
            send()
            return .handled
          }
          .padding(6)
          .frame(minHeight: 52, maxHeight: 140)

        if draft.isEmpty {
          Text("Ask the agent to change or explain something…")
            .foregroundStyle(.tertiary)
            .padding(.leading, 11)
            .padding(.top, 12)
            .allowsHitTesting(false)
        }
      }
      .background(
        Color(nsColor: .textBackgroundColor),
        in: RoundedRectangle(cornerRadius: 8))
      .overlay(
        RoundedRectangle(cornerRadius: 8)
          .stroke(Color(nsColor: .separatorColor), lineWidth: 1))

      VStack(spacing: 8) {
        if session.isRunning {
          Button("Stop", role: .cancel) { session.cancel() }
        } else {
          Button("Send") { send() }
            .keyboardShortcut(.return, modifiers: [.command])
            .disabled(!session.canSend || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
      }
    }
    .padding(.horizontal, horizontalPadding)
    .padding(.vertical, 12)
  }

  // MARK: - Actions

  private func send() {
    let text = draft
    guard session.canSend,
      !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { return }
    draft = ""
    session.send(text)
    composerFocused = true
  }

  private func chooseProject() {
    let selection = ModalPresentation.run { () -> URL? in
      let panel = NSOpenPanel()
      panel.canChooseFiles = false
      panel.canChooseDirectories = true
      panel.canCreateDirectories = false
      panel.allowsMultipleSelection = false
      panel.message = "Choose the project folder the agent can read and change"
      panel.prompt = "Choose"
      panel.directoryURL = session.workspaceURL
      return panel.runModal() == .OK ? panel.url : nil
    }
    if let selection {
      session.setWorkspace(selection)
    }
  }

  private func raiseContext(to target: ContextTier) {
    guard let model = currentModel else { return }
    UserSettings.setSelectedCtxTier(target, for: model.id)
    _ = ModelManager.shared.updateModelsFile()
    if LlamaServer.shared.isActive(model: model) {
      LlamaServer.shared.reload()
    }
    models = ModelManager.shared.downloadedModels
    refreshContextNotice()
  }

  /// Local models default to a 4k context, which is too small for a tool loop
  /// once the system prompt and tool schemas are in. Offer the largest tier up
  /// to 32k that this Mac can actually fit.
  private func refreshContextNotice() {
    guard let model = currentModel, let current = model.effectiveCtxTier,
      current < .k16
    else {
      contextNotice = nil
      return
    }
    let supported = model.supportedContextTiers
    let target = supported.last { $0 <= .k32 } ?? supported.last
    let pinned = UserModelOverrides.overriddenCtxSize(for: model.id) != nil
    contextNotice = ContextNotice(
      current: current, target: pinned || target == current ? nil : target, pinned: pinned)
  }

  private func contextBanner(_ notice: ContextNotice) -> some View {
    HStack(spacing: 10) {
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundStyle(.orange)

      VStack(alignment: .leading, spacing: 2) {
        Text("This model is set to \(notice.current.shortLabel) context")
          .font(.callout.weight(.medium))
        Text(
          notice.target != nil
            ? "Agent work needs more room. Raising it restarts the server for this model."
            : notice.pinned
              ? "That's tight for agent work, and models.user.ini pins this model's context."
              : "That's tight for agent work, and a larger tier doesn't fit in memory right now."
        )
        .font(.caption)
        .foregroundStyle(.secondary)
      }

      Spacer()

      if let target = notice.target {
        Button("Use \(target.shortLabel)") { raiseContext(to: target) }
          .controlSize(.small)
      }
    }
    .padding(.horizontal, horizontalPadding)
    .padding(.vertical, 8)
    .background(Color.orange.opacity(0.08))
  }

  private func refreshModels() {
    models = ModelManager.shared.downloadedModels
    if !models.contains(where: { $0.id == session.modelID }) {
      session.setModel(models.first?.id ?? "")
    }
    refreshContextNotice()
  }

  private func abbreviatedPath(_ url: URL, maxLength: Int) -> String {
    let path = url.path
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    let abbreviated = path.hasPrefix(home) ? "~" + path.dropFirst(home.count) : path
    guard abbreviated.count > maxLength else { return abbreviated }
    let keep = maxLength - 1
    return "\(abbreviated.prefix(keep - keep / 2))…\(abbreviated.suffix(keep / 2))"
  }
}

// MARK: - Entries

private struct AgentEntryView: View {
  let entry: AgentSession.TranscriptEntry

  var body: some View {
    switch entry.kind {
    case .user:
      HStack {
        Spacer(minLength: 60)
        Text(entry.text)
          .textSelection(.enabled)
          .padding(.horizontal, 12)
          .padding(.vertical, 8)
          .background(
            Color.accentColor.opacity(0.16),
            in: RoundedRectangle(cornerRadius: 10))
      }
    case .assistant:
      if entry.isPending && entry.text.isEmpty {
        HStack(spacing: 6) {
          ProgressView().controlSize(.small)
          Text("Thinking…").foregroundStyle(.secondary)
        }
        .font(.callout)
      } else {
        MarkdownText(text: entry.text)
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    case .reasoning:
      ReasoningView(text: entry.text)
    case .toolCall:
      ToolCallView(entry: entry)
    case .toolResult:
      ToolResultView(entry: entry)
    case .notice:
      Label(entry.text, systemImage: "info.circle")
        .font(.callout)
        .foregroundStyle(.secondary)
    case .error:
      Label(entry.text, systemImage: "exclamationmark.triangle.fill")
        .font(.callout)
        .foregroundStyle(.red)
        .textSelection(.enabled)
    }
  }
}

private struct MarkdownText: View {
  let text: String

  var body: some View {
    if let attributed = try? AttributedString(markdown: text) {
      Text(attributed)
    } else {
      Text(text)
    }
  }
}

private struct ReasoningView: View {
  let text: String
  @State private var expanded = false

  var body: some View {
    DisclosureGroup(isExpanded: $expanded) {
      Text(text)
        .font(.callout)
        .foregroundStyle(.secondary)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 4)
    } label: {
      Label("Reasoning", systemImage: "brain")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
  }
}

private struct ToolCallView: View {
  let entry: AgentSession.TranscriptEntry
  @State private var expanded = false

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 6) {
        Image(systemName: "wrench.and.screwdriver")
          .foregroundStyle(.secondary)
        Text(entry.title ?? "Tool")
          .font(.system(.callout, design: .monospaced).weight(.medium))
        Text(entry.text)
          .font(.callout)
          .foregroundStyle(.secondary)
          .lineLimit(1)
        Spacer(minLength: 0)
        if entry.detail != nil {
          Button(expanded ? "Hide" : "Details") { expanded.toggle() }
            .buttonStyle(.link)
            .font(.caption)
        }
      }
      if expanded, let detail = entry.detail {
        Text(detail)
          .font(.system(size: 11, design: .monospaced))
          .textSelection(.enabled)
          .padding(8)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(
            Color(nsColor: .textBackgroundColor),
            in: RoundedRectangle(cornerRadius: 6))
      }
    }
    .padding(10)
    .background(
      Color(nsColor: .controlBackgroundColor),
      in: RoundedRectangle(cornerRadius: 8))
  }
}

private struct ToolResultView: View {
  let entry: AgentSession.TranscriptEntry
  @State private var expanded = false

  private var isLong: Bool { entry.text.count > 900 }

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 6) {
        Image(systemName: entry.isError ? "xmark.circle.fill" : "checkmark.circle.fill")
          .foregroundStyle(entry.isError ? .red : .green)
        Text(entry.title ?? "Result")
          .font(.system(.callout, design: .monospaced).weight(.medium))
        Spacer()
        if isLong {
          Button(expanded ? "Collapse" : "Show all") { expanded.toggle() }
            .buttonStyle(.link)
            .font(.caption)
        }
      }
      Text(entry.text)
        .font(.system(size: 11, design: .monospaced))
        .foregroundStyle(entry.isError ? .red : .primary)
        .textSelection(.enabled)
        .lineLimit(expanded ? nil : 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(10)
    .background(
      Color(nsColor: .textBackgroundColor),
      in: RoundedRectangle(cornerRadius: 8))
    .overlay(
      RoundedRectangle(cornerRadius: 8)
        .stroke(Color(nsColor: .separatorColor), lineWidth: 1))
  }
}

private struct AgentApprovalCard: View {
  let approval: AgentSession.PendingApproval
  let session: AgentSession
  @State private var remember = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label("Allow \(approval.title)?", systemImage: "exclamationmark.shield.fill")
        .font(.callout.weight(.semibold))
        .foregroundStyle(.orange)

      Text(approval.detail)
        .font(.system(size: 11, design: .monospaced))
        .textSelection(.enabled)
        .lineLimit(8)
        .frame(maxWidth: .infinity, alignment: .leading)

      HStack(spacing: 10) {
        Toggle("Always allow \(approval.toolName) this session", isOn: $remember)
          .toggleStyle(.checkbox)
          .font(.callout)

        Spacer()

        Button("Deny", role: .cancel) {
          session.resolveApproval(allow: false, remember: false)
        }

        Button("Allow") {
          session.resolveApproval(allow: true, remember: remember)
        }
        .keyboardShortcut(.defaultAction)
      }
    }
    .padding(.horizontal, 18)
    .padding(.vertical, 10)
    .background(Color.orange.opacity(0.08))
  }
}
