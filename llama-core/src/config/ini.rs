//! Pure port of the INI parsing/merging in
//! `Llama/Server/UserModelOverrides.swift`. File I/O, seeding and the
//! suspension flag stay in Swift; everything here is a pure function over text.

/// Filename of the user-owned file.
pub const FILENAME: &str = "models.user.ini";

/// One `[section]` and its keys, in the order they were written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub pairs: Vec<(String, String)>,
}

impl Section {
    pub fn new(name: impl Into<String>) -> Section {
        Section {
            name: name.into(),
            pairs: Vec::new(),
        }
    }

    /// Sets `key`, replacing in place or appending.
    pub fn set(&mut self, key: &str, value: &str) {
        if let Some(pair) = self.pairs.iter_mut().find(|(k, _)| k == key) {
            pair.1 = value.to_string();
        } else {
            self.pairs.push((key.to_string(), value.to_string()));
        }
    }

    /// Removes `key` if present.
    pub fn remove(&mut self, key: &str) {
        self.pairs.retain(|(k, _)| k != key);
    }

    /// The value for `key`, if present.
    pub fn value(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn serialized(&self) -> String {
        let mut out = format!("[{}]\n", self.name);
        for (key, value) in &self.pairs {
            out += &format!("{key} = {value}\n");
        }
        out + "\n"
    }
}

/// Minimal INI parse: `[section]` headers and `key = value` lines.
pub fn parse(text: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();

    for raw_line in text.split(['\n', '\r']) {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            let name = line[1..line.len() - 1].trim();
            if !name.is_empty() {
                sections.push(Section::new(name));
            }
            continue;
        }

        // `value` may itself contain `=`, so split on the first separator only.
        let Some(sep) = line.find('=') else {
            continue;
        };
        if sections.is_empty() {
            continue;
        }
        let key = line[..sep].trim();
        let value = line[sep + 1..].trim();
        if key.is_empty() {
            continue;
        }
        let last = sections.len() - 1;
        sections[last].set(key, value);
    }

    sections
}

/// Applies user overrides to app-generated sections. Merges matching sections
/// per-key and appends sections the app didn't generate. When `suspended`,
/// returns `generated` unchanged.
pub fn apply(generated: &[Section], overrides: &[Section], suspended: bool) -> Vec<Section> {
    if suspended {
        return generated.to_vec();
    }

    let mut merged = generated.to_vec();
    let mut index_by_name: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (index, section) in merged.iter().enumerate() {
        index_by_name.insert(section.name.clone(), index);
    }

    for over in overrides {
        if let Some(&index) = index_by_name.get(&over.name) {
            for (key, value) in &over.pairs {
                merged[index].set(key, value);
            }
        } else {
            index_by_name.insert(over.name.clone(), merged.len());
            merged.push(over.clone());
        }
    }

    merged
}
