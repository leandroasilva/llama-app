//! Pure port of `Llama/Server/ModelAliases.swift`. The alias store
//! (`UserSettings.modelAliases`) and the parsed override file are injected.

use crate::config::ini::{Section, FILENAME};

/// A stable name that API clients can send as `"model"` in place of a model id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Alias {
    pub name: String,
    pub model_id: String,
}

/// The preset key the router reads aliases from.
pub const PRESET_KEY: &str = "alias";

/// Why `name` can't be saved as an alias, or `None` if it can.
///
/// - `aliases`: the user's current aliases.
/// - `replacing`: the alias being edited, whose own name doesn't count as taken.
/// - `override_names`: section names in `models.user.ini` (the `current` store).
pub fn validation_error(
    name: &str,
    aliases: &[Alias],
    replacing: Option<&Alias>,
    override_names: &[String],
) -> Option<String> {
    if name.is_empty() {
        return Some("Enter a name.".to_string());
    }
    if !name.chars().all(is_allowed) {
        return Some("Use lowercase letters, digits, dots, dashes, or underscores.".to_string());
    }
    let taken = aliases.iter().filter(|a| replacing != Some(*a));
    if taken.into_iter().any(|a| a.name.as_str() == name) {
        return Some(format!("There's already an alias named {name}."));
    }
    if override_names.iter().any(|n| n == name) {
        return Some(format!("{FILENAME} defines a model named {name}."));
    }
    None
}

/// Characters an alias may use.
fn is_allowed(c: char) -> bool {
    matches!(c, 'a'..='z' | '0'..='9' | '.' | '_' | '-')
}

/// Sets the `alias` key on a model's generated section. No-op when `names` is
/// empty.
pub fn set_alias(section: &mut Section, names: &[String]) {
    if names.is_empty() {
        return;
    }
    section.set(PRESET_KEY, &names.join(","));
}

/// Drops any alias the router would refuse, from the final merged sections.
/// First claim wins.
pub fn remove_conflicts(sections: &[Section]) -> Vec<Section> {
    let model_names: std::collections::HashSet<&str> =
        sections.iter().map(|s| s.name.as_str()).collect();
    let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();

    sections
        .iter()
        .map(|section| {
            let Some(value) = section.value(PRESET_KEY) else {
                return section.clone();
            };

            let mut kept: Vec<String> = Vec::new();
            for alias in value.split(',').map(str::trim).filter(|a| !a.is_empty()) {
                if model_names.contains(alias) || claimed.contains(alias) {
                    continue;
                }
                claimed.insert(alias.to_string());
                kept.push(alias.to_string());
            }

            let mut section = section.clone();
            if kept.is_empty() {
                section.remove(PRESET_KEY);
            } else {
                section.set(PRESET_KEY, &kept.join(","));
            }
            section
        })
        .collect()
}
