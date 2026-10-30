import AppKit

/// Compact live runtime readout shown whenever a model is resident. Throughput
/// comes from llama.cpp's own Prometheus gauges; memory comes from the server
/// process's macOS physical-footprint sample.
final class RuntimeMetricsView: ItemView {
  private unowned let server: LlamaServer
  private let modelLabel = Theme.primaryLabel()
  private let metricsLabel = Theme.secondaryLabel()

  init(server: LlamaServer) {
    self.server = server
    super.init(frame: .zero)

    modelLabel.maximumNumberOfLines = 1
    modelLabel.lineBreakMode = .byTruncatingTail
    metricsLabel.maximumNumberOfLines = 1
    metricsLabel.lineBreakMode = .byTruncatingTail
    metricsLabel.textColor = Theme.Colors.textSecondary

    let stack = NSStackView(views: [modelLabel, metricsLabel])
    stack.orientation = .vertical
    stack.alignment = .leading
    stack.spacing = 2
    contentView.addSubview(stack)
    stack.pinToSuperview()

    widthAnchor.constraint(equalToConstant: Layout.menuWidth).isActive = true
    heightAnchor.constraint(equalToConstant: 40).isActive = true
    refresh()
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  override var highlightEnabled: Bool { false }

  func refresh() {
    guard let modelId = server.runtimeModelId else {
      modelLabel.stringValue = "Runtime"
      metricsLabel.stringValue = "No model loaded"
      return
    }

    modelLabel.stringValue = ModelIdParser.plainName(modelId, showTags: false)
    let memory = Format.memory(mb: server.currentPhysicalFootprintBytes / 1_048_576)
    guard let metrics = server.runtimeMetrics else {
      metricsLabel.stringValue = "Memory \(memory) · collecting metrics…"
      return
    }

    if metrics.requestsProcessing > 0 {
      metricsLabel.stringValue = String(
        format: "RAM %@ · %.0f tok/s · prompt %.0f tok/s",
        memory, metrics.generationTokensPerSecond, metrics.promptTokensPerSecond)
    } else {
      if let lastLoad = server.mostRecentLoadMetric, lastLoad.modelId == modelId {
        let duration = String(format: "%.1fs", Double(lastLoad.durationMs) / 1_000)
        let peak = Format.memory(mb: lastLoad.peakPhysicalFootprintBytes / 1_048_576)
        metricsLabel.stringValue = "RAM \(memory) · idle · load \(duration) · peak \(peak)"
      } else {
        metricsLabel.stringValue = "RAM \(memory) · idle"
      }
    }
    toolTip = metrics.requestsProcessing > 0
      ? "\(metrics.requestsProcessing) request(s) processing"
      : "llama.cpp runtime metrics"
  }
}
