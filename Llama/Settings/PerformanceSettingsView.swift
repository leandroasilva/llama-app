import SwiftUI

/// Controls how much context models get and how much memory llama.cpp spends
/// on it. Global defaults live in the `[*]` section of `models.ini`; a model
/// can opt into its own mirror of these values.
struct PerformanceSettingsView: View {
  @State private var contextPolicy = UserSettings.contextLengthPolicy
  @State private var config = UserSettings.performanceConfig
  @State private var useFullWorkingSet = UserSettings.useFullWorkingSet
  @State private var models: [Model] = []
  @State private var selectedModelID = ""
  @State private var isOverriding = false
  @State private var modelConfig = PerformanceConfig.default

  private var selectedModel: Model? {
    models.first { $0.id == selectedModelID }
  }

  var body: some View {
    Form {
      Section {
        SettingRow(
          title: "Default context length",
          description:
            "Largest that fits uses the most memory the Mac can spare. Smallest (4K) leaves more room free."
        ) {
          Picker(
            "Default context length",
            selection: Binding(
              get: { contextPolicy },
              set: { newValue in
                contextPolicy = newValue
                UserSettings.contextLengthPolicy = newValue
              })
          ) {
            ForEach(ContextLengthPolicy.allCases) { policy in
              Text(policy.label).tag(policy)
            }
          }
          .labelsHidden()
          .frame(maxWidth: 200)
        }
      }

      Section {
        memoryControls(globalBinding)
      }

      Section {
        SettingRow(
          title: "Use full GPU working set",
          description:
            "Spends llama.cpp's safety margin on model memory. Can make a model that just misses the default fit, at the risk of a failed load."
        ) {
          Toggle(
            "",
            isOn: Binding(
              get: { useFullWorkingSet },
              set: { newValue in
                useFullWorkingSet = newValue
                UserSettings.useFullWorkingSet = newValue
              })
          )
          .labelsHidden()
        }

        if useFullWorkingSet {
          SettingCaution(
            text:
              "Without the safety margin, a model can load on paper and still fail, especially while another model is briefly resident."
          )
        }
      }

      Section {
        SettingRow(
          title: "Model",
          description:
            "Choose a model to give it a fixed context length or its own memory settings."
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

        if let model = selectedModel {
          contextRow(for: model)

          SettingRow(
            title: "Custom memory settings",
            description: "Use this model's own memory settings instead of the defaults above."
          ) {
            Toggle(
              "",
              isOn: Binding(
                get: { isOverriding },
                set: { setOverride($0) })
            )
            .labelsHidden()
          }

          if isOverriding {
            memoryControls(modelBinding)
          }
        } else {
          Text("No installed models.")
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
    .onChange(of: selectedModelID) { _, _ in
      syncModelSelection()
    }
  }

  @ViewBuilder
  private func memoryControls(_ config: Binding<PerformanceConfig>) -> some View {
    SettingRow(
      title: "Flash attention",
      description: "Lower memory use and faster attention when the model supports it."
    ) {
      Toggle("", isOn: config.flashAttention)
        .labelsHidden()
    }

    SettingRow(
      title: "KV cache K",
      description: "Quantizing the K cache saves memory. q8_0 is nearly lossless; f16 uses the most."
    ) {
      Picker("KV cache K", selection: config.kvCacheTypeK) {
        ForEach(KVCacheType.allCases) { type in
          Text(type.label).tag(type)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 110)
    }

    SettingRow(
      title: "KV cache V",
      description: "The V cache can usually use the same type as K."
    ) {
      Picker("KV cache V", selection: config.kvCacheTypeV) {
        ForEach(KVCacheType.allCases) { type in
          Text(type.label).tag(type)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 110)
    }

    SettingRow(
      title: "Parallel slots",
      description: "One slot uses the least memory and matches a single local conversation."
    ) {
      Picker("Parallel slots", selection: config.parallel) {
        ForEach(Array(PerformanceConfig.parallelRange), id: \.self) { value in
          Text("\(value)").tag(value)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 70)
    }

    SettingRow(
      title: "Prompt batch",
      description: "How many tokens are processed at once. Larger batches use more memory."
    ) {
      Picker("Prompt batch", selection: config.batchSize) {
        ForEach(PerformanceConfig.batchOptions, id: \.self) { value in
          Text("\(value)").tag(value)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 90)
    }

    SettingRow(
      title: "Micro-batch",
      description: "The per-step batch. Keep this at or below the prompt batch."
    ) {
      Picker("Micro-batch", selection: config.ubatchSize) {
        ForEach(PerformanceConfig.ubatchOptions.filter { $0 <= config.wrappedValue.batchSize },
          id: \.self
        ) { value in
          Text("\(value)").tag(value)
        }
      }
      .labelsHidden()
      .frame(maxWidth: 90)
    }

    SettingRow(
      title: "Keep KV cache in RAM",
      description: "Offloads the cache from the GPU. Saves GPU memory on discrete setups; on Apple silicon it usually costs speed."
    ) {
      Toggle("", isOn: config.noKVOffload)
        .labelsHidden()
    }
  }

  @ViewBuilder
  private func contextRow(for model: Model) -> some View {
    if let overridden = UserModelOverrides.overriddenCtxSize(for: model.id) {
      SettingRow(
        title: "Context length",
        description:
          "Pinned in \(UserModelOverrides.filename). Remove ctx-size there to use this picker again."
      ) {
        Text(Format.ctxOverride(overridden))
          .font(.system(.callout, design: .monospaced))
      }
    } else {
      SettingRow(
        title: "Context length",
        description: "Automatic follows the default above. A fixed size overrides it for this model."
      ) {
        Picker("Context length", selection: contextSelection(for: model)) {
          Text("Automatic (\(model.effectiveCtxTier?.shortLabel ?? "4k"))")
            .tag(0)

          ForEach(contextOptions(for: model)) { tier in
            Text(tier.label).tag(tier.rawValue)
          }
        }
        .labelsHidden()
        .frame(maxWidth: 190)
      }
    }
  }

  private var globalBinding: Binding<PerformanceConfig> {
    Binding(
      get: { config },
      set: { newValue in
        let normalized = newValue.normalized
        config = normalized
        UserSettings.performanceConfig = normalized
      })
  }

  private var modelBinding: Binding<PerformanceConfig> {
    Binding(
      get: { modelConfig },
      set: { newValue in
        let normalized = newValue.normalized
        modelConfig = normalized
        guard !selectedModelID.isEmpty else { return }
        UserSettings.setPerformanceConfig(normalized, for: selectedModelID)
      })
  }

  private func contextSelection(for model: Model) -> Binding<Int> {
    Binding(
      get: { UserSettings.selectedCtxTier(for: model.id)?.rawValue ?? 0 },
      set: { rawValue in
        let tier = rawValue == 0 ? nil : ContextTier(rawValue: rawValue)
        UserSettings.setSelectedCtxTier(tier, for: model.id)
      })
  }

  /// Tiers this Mac can run, plus a previously saved tier that no longer fits
  /// so the picker can still show the current choice.
  private func contextOptions(for model: Model) -> [ContextTier] {
    var tiers = model.supportedContextTiers
    if let selected = UserSettings.selectedCtxTier(for: model.id), !tiers.contains(selected) {
      tiers.append(selected)
      tiers.sort()
    }
    return tiers
  }

  private func setOverride(_ enabled: Bool) {
    guard let model = selectedModel else {
      isOverriding = false
      return
    }

    if enabled {
      let starting =
        UserSettings.hasPerformanceOverride(for: model.id)
        ? UserSettings.performanceConfig(for: model.id)
        : config
      modelConfig = starting
      isOverriding = true
      UserSettings.setPerformanceConfig(starting, for: model.id)
    } else {
      isOverriding = false
      UserSettings.setPerformanceConfig(nil, for: model.id)
      modelConfig = config
    }
  }

  private func refreshModels() {
    models = ModelManager.shared.downloadedModels
    if !models.contains(where: { $0.id == selectedModelID }) {
      selectedModelID = models.first?.id ?? ""
    }
    syncModelSelection()
  }

  private func syncModelSelection() {
    guard let model = selectedModel else {
      isOverriding = false
      return
    }
    isOverriding = UserSettings.hasPerformanceOverride(for: model.id)
    modelConfig = UserSettings.performanceConfig(for: model.id)
  }
}
