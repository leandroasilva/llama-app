//! Small path helpers shared by the ports. Mirrors Foundation's
//! `lastPathComponent` closely enough for repo-relative GGUF paths.

/// The final path component, ignoring a trailing slash. `"a/b/c.gguf"` ->
/// `"c.gguf"`, `"a.gguf"` -> `"a.gguf"`, `"dir/"` -> `"dir"`.
pub fn last_path_component(path: &str) -> &str {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(i) => &trimmed[i + 1..],
        None => trimmed,
    }
}

/// The directory components of a repo-relative path, mirroring Swift's
/// `path.split(separator: "/").dropLast()` (which omits empty subsequences):
/// `"Q4_K_M/a.gguf"` -> `["Q4_K_M"]`, `"a.gguf"` -> `[]`.
pub fn dir_components(path: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    parts.pop();
    parts
}
