import Foundation

/// A stable name that API clients can send as `"model"` in place of a model
/// id, pointed at one installed model.
///
/// The point is portability: code written against `"model": "code"` runs
/// unchanged on every Mac, and each Mac decides which model answers to
/// `code` -- a big one where there's memory for it, a small one where there
/// isn't. So the mapping is deliberately per-device and never synced.
struct ModelAlias: Equatable, Hashable {
  var name: String
  var modelId: String
}

/// Turns the user's aliases into `alias` keys in `models.ini`, and keeps
/// them from ever producing a file the server won't start with.
///
/// The router resolves an alias to its model on every request (and lists it
/// under the model's `aliases` in `/models`), so there's no server-side
/// feature to build here -- only config to write.
enum ModelAliases {
  /// The preset key the router reads aliases from (`--alias`, comma-separated).
  static let presetKey = "alias"

  /// The names pointed at `modelId`, in the order the user added them.
  static func names(for modelId: String) -> [String] {
    UserSettings.modelAliases.filter { $0.modelId == modelId }.map(\.name)
  }

  /// Why `name` can't be saved as an alias, or nil if it can.
  ///
  /// Validation (allowed characters, uniqueness, collision with a user-defined
  /// model name) lives in the Rust core (`aliases.validationError`); the alias
  /// store and the parsed override file are injected as inputs.
  ///
  /// - Parameter replacing: the alias being edited, whose own name doesn't
  ///   count as taken.
  static func validationError(for name: String, replacing: ModelAlias? = nil) -> String? {
    var args: [String: Any] = [
      "name": name,
      "aliases": UserSettings.modelAliases.map { ["name": $0.name, "modelId": $0.modelId] },
      "overrideNames": UserModelOverrides.current.map(\.name),
    ]
    if let replacing {
      args["replacing"] = ["name": replacing.name, "modelId": replacing.modelId]
    }
    return LlamaCore.callValue("aliases.validationError", "error", args)
  }

  /// The aliases a `models.user.ini` entry sets for `modelId`, if it sets any.
  ///
  /// Overrides merge per key and the user always wins, so such an entry
  /// replaces the aliases set in the app rather than adding to them. The UI
  /// uses this to say so, instead of showing an alias that isn't in effect.
  static func overridden(for modelId: String) -> String? {
    UserModelOverrides.overriddenValue(presetKey, for: modelId)
  }

  /// Sets the `alias` key on a model's generated section. Applied before user
  /// overrides, so a user's own `alias` key replaces it like any other key.
  static func apply(to section: inout UserModelOverrides.Section) {
    let names = names(for: section.name)
    guard !names.isEmpty else { return }
    section.set(presetKey, names.joined(separator: ","))
  }

  /// Drops any alias the router would refuse, from the final merged sections.
  ///
  /// The router throws on an alias that matches a model name or another
  /// model's alias, and on a fresh start that takes the whole server down --
  /// every model, over one name. The app's own aliases can't conflict (the
  /// UI validates them), but a `models.user.ini` edit can introduce a clash
  /// at any time, and a server that won't start is a much worse outcome than
  /// one alias not resolving. First claim wins: generated sections come
  /// before user-only ones, so an alias set in the app keeps working.
  ///
  /// The merge is done by the Rust core (`aliases.removeConflicts`); sections
  /// are exchanged as `{name, pairs:[{key,value}]}` to preserve key order.
  static func removeConflicts(from sections: [UserModelOverrides.Section])
    -> [UserModelOverrides.Section]
  {
    let payload = sections.map { section -> [String: Any] in
      [
        "name": section.name,
        "pairs": section.pairs.map { ["key": $0.key, "value": $0.value] },
      ]
    }
    guard
      let out: [[String: Any]] = LlamaCore.callValue(
        "aliases.removeConflicts", "sections", ["sections": payload])
    else { return sections }

    return out.map { obj in
      let pairs = (obj["pairs"] as? [[String: Any]] ?? []).map {
        (key: $0["key"] as? String ?? "", value: $0["value"] as? String ?? "")
      }
      return UserModelOverrides.Section(name: obj["name"] as? String ?? "", pairs: pairs)
    }
  }
}
