import AppKit

final class FooterView: ItemView {
  private let onCheckForUpdates: () -> Void
  private let onOpenSettings: () -> Void
  private let onQuit: () -> Void
  /// Build of the `llama` binary actually in use, or nil if not yet known.
  private let llamaVersion: String?
  /// Where that binary comes from -- unmanaged installs get a marker ("· brew"
  /// or "· ext") so a stale version isn't mistaken for a bug.
  private let llamaOrigin: LlamaBinaries.Origin
  /// Version Sparkle is holding back as a gentle reminder, or nil if none. Shown
  /// beside the running version ("0.0.8 → 0.0.9") because a background app has
  /// no dock badge to point at, and its update alert can open behind other apps.
  private var updateVersion: String?

  /// The running version, clickable to run an update check. A plain label (not
  /// an NSButton) so its text starts at the exact content edge; a borderless
  /// button's cell adds a ~2px inset that left it looking misaligned against the
  /// header and model rows. Clickable via a gesture, matching the menu's other
  /// inline links (WebUI, "models").
  private lazy var versionLabel: NSTextField = {
    let label = Theme.secondaryLabel()
    label.isSelectable = false
    label.translatesAutoresizingMaskIntoConstraints = false
    label.addGestureRecognizer(
      NSClickGestureRecognizer(target: self, action: #selector(checkForUpdatesClicked)))
    return label
  }()

  init(
    llamaVersion: String?,
    llamaOrigin: LlamaBinaries.Origin,
    updateVersion: String?,
    onCheckForUpdates: @escaping () -> Void,
    onOpenSettings: @escaping () -> Void,
    onQuit: @escaping () -> Void
  ) {
    self.llamaVersion = llamaVersion
    self.llamaOrigin = llamaOrigin
    self.updateVersion = updateVersion
    self.onCheckForUpdates = onCheckForUpdates
    self.onOpenSettings = onOpenSettings
    self.onQuit = onQuit
    super.init(frame: .zero)
    setup()
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override var highlightEnabled: Bool { false }

  override var intrinsicContentSize: NSSize {
    NSSize(width: Layout.menuWidth, height: 28)
  }

  private func setup() {
    refreshVersionLabel()
    // Llama Version Label -- the build of the binary actually being driven.
    // Unmanaged installs get a marker ("· brew" for Homebrew, "· ext" for
    // anything else), since the app won't update them and their version can
    // legitimately trail the pin.
    let marker: String
    switch llamaOrigin {
    case .managed: marker = ""
    case .brew: marker = " · brew"
    case .external: marker = " · ext"
    }
    let llamaText = llamaVersion.map { " · llama.cpp \($0)\(marker)" } ?? " · llama.cpp"
    let llamaLabel = Theme.tertiaryLabel(llamaText)
    llamaLabel.translatesAutoresizingMaskIntoConstraints = false

    // Settings Button
    let settingsButton = FooterButton(
      title: "Settings", target: self, action: #selector(openSettingsClicked))
    settingsButton.translatesAutoresizingMaskIntoConstraints = false

    // Quit Button
    let quitButton = FooterButton(title: "Quit", target: self, action: #selector(quitClicked))
    quitButton.translatesAutoresizingMaskIntoConstraints = false

    contentView.addSubview(versionLabel)
    contentView.addSubview(llamaLabel)
    contentView.addSubview(settingsButton)
    contentView.addSubview(quitButton)

    NSLayoutConstraint.activate([
      // Left side
      versionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
      versionLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),

      llamaLabel.leadingAnchor.constraint(equalTo: versionLabel.trailingAnchor),
      llamaLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),

      // Right side
      quitButton.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
      quitButton.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),

      settingsButton.trailingAnchor.constraint(equalTo: quitButton.leadingAnchor, constant: -5),
      settingsButton.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
    ])
  }

  @objc private func checkForUpdatesClicked() { onCheckForUpdates() }
  @objc private func openSettingsClicked() { onOpenSettings() }
  @objc private func quitClicked() { onQuit() }

  /// Records the update Sparkle is holding back (nil when there is none) and
  /// re-renders the version label. Lets the menu controller reflect a new update
  /// without rebuilding an already-open menu.
  func setUpdateVersion(_ version: String?) {
    guard updateVersion != version else { return }
    updateVersion = version
    refreshVersionLabel()
  }

  private func refreshVersionLabel() {
    let color = updateVersion == nil ? Theme.Colors.textPrimary : NSColor.controlAccentColor
    versionLabel.attributedStringValue = NSAttributedString(
      string: appVersionText,
      attributes: Theme.secondaryAttributes(color: color)
    )
    versionLabel.toolTip = updateVersion.map {
      "New version \($0) available. Click to check for updates."
    } ?? "Click to check for updates"
  }

  private var appVersionText: String {
    let running: String
    #if DEBUG
      running = "dev"
    #else
      let version = AppInfo.shortVersion
      // Unreleased builds report 0.0.0, so pair it with the build number rather
      // than printing a bare number that reads like a version of its own.
      running = version == "0.0.0" ? "0.0.0 (\(AppInfo.buildNumber))" : version
    #endif
    guard let updateVersion else { return running }
    return "\(running) → \(updateVersion)"
  }
}

/// A simple bordered button matching the footer style
private class FooterButton: NSButton {
  init(title: String, target: AnyObject?, action: Selector) {
    super.init(frame: .zero)
    self.attributedTitle = NSAttributedString(
      string: title,
      attributes: Theme.secondaryAttributes(color: Theme.Colors.textSecondary)
    )
    self.target = target
    self.action = action
    self.bezelStyle = .inline
    self.isBordered = false  // We draw our own border
    self.wantsLayer = true
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override var wantsUpdateLayer: Bool { true }

  override var intrinsicContentSize: NSSize {
    let size = super.intrinsicContentSize
    return NSSize(width: size.width + 8, height: size.height + 4)
  }

  override func updateLayer() {
    layer?.cornerRadius = 5
    layer?.borderWidth = 1
    // Use Theme.Colors.border instead of .separatorColor because CALayers don't support vibrancy.
    // See Theme.swift for details.
    layer?.setBorderColor(Theme.Colors.border, in: self)

    let bgColor: NSColor = isHighlighted ? Theme.Colors.subtleBackground : .clear
    layer?.setBackgroundColor(bgColor, in: self)
  }

  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    needsDisplay = true
  }

  override var isHighlighted: Bool {
    didSet { needsDisplay = true }
  }
}
