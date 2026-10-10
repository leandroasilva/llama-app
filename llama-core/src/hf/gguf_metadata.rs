//! Minimal reader for the GGUF key-value header.
//!
//! GGUF files open with a small self-describing metadata section (magic,
//! version, tensor count, then typed key-value pairs) before the tensor data.
//! We parse just that section to answer questions the filename can't -- starting
//! with embedded MTP heads, which some publishers (unsloth) ship without any
//! filename marker. The file is memory-mapped, so only the header pages are ever
//! touched even on multi-GB models.
//!
//! The parser is deliberately fail-soft: any structural surprise (unknown
//! version, truncated header, unknown value type) returns `None` rather than
//! erroring -- callers fall back to their filename heuristics. Misreading a
//! model must never block install or scan.
//!
//! Ported byte-for-byte from `Llama/HF/GGUFMetadata.swift`.

use memmap2::Mmap;
use std::fs::File;

/// GGUF metadata value types, per the spec's `gguf_metadata_value_type`.
/// Raw values are the on-disk type ids.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl ValueType {
    fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            0 => ValueType::Uint8,
            1 => ValueType::Int8,
            2 => ValueType::Uint16,
            3 => ValueType::Int16,
            4 => ValueType::Uint32,
            5 => ValueType::Int32,
            6 => ValueType::Float32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::Uint64,
            11 => ValueType::Int64,
            12 => ValueType::Float64,
            _ => return None,
        })
    }
}

/// The result of scanning the KV section: the two nil-ish outcomes stay apart.
enum Scan<V> {
    /// The header couldn't be parsed at all.
    Unparseable,
    /// The header parsed cleanly but the key isn't there.
    Absent,
    /// The key was found; its value was read with the supplied closure.
    Present(V),
}

/// Bounds-checked little-endian reader over the mapped file.
struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, offset: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(n)?;
        if end <= self.data.len() {
            let s = &self.data[self.offset..end];
            self.offset = end;
            Some(s)
        } else {
            None
        }
    }

    fn read_u8(&mut self) -> Option<u8> {
        self.read_bytes(1).map(|b| b[0])
    }

    fn read_i8(&mut self) -> Option<i8> {
        self.read_bytes(1).map(|b| b[0] as i8)
    }

    fn read_u16(&mut self) -> Option<u16> {
        self.read_bytes(2).map(|b| u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Option<i16> {
        self.read_bytes(2).map(|b| i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Option<u32> {
        self.read_bytes(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Option<i32> {
        self.read_bytes(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Option<u64> {
        self.read_bytes(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_i64(&mut self) -> Option<i64> {
        self.read_bytes(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_string(&mut self) -> Option<String> {
        let length = self.read_u64()?;
        if length >= 1_000_000 {
            return None;
        }
        let bytes = self.read_bytes(length as usize)?;
        String::from_utf8(bytes.to_vec()).ok()
    }

    fn skip(&mut self, count: usize) -> bool {
        match self.offset.checked_add(count) {
            Some(end) if end <= self.data.len() => {
                self.offset = end;
                true
            }
            _ => false,
        }
    }

    /// Reads a numeric value of `ty` as `i64`; `None` for non-numeric types.
    fn read_int_value(&mut self, ty: ValueType) -> Option<i64> {
        match ty {
            ValueType::Uint8 => self.read_u8().map(i64::from),
            ValueType::Int8 => self.read_i8().map(i64::from),
            ValueType::Uint16 => self.read_u16().map(i64::from),
            ValueType::Int16 => self.read_i16().map(i64::from),
            ValueType::Uint32 => self.read_u32().map(i64::from),
            ValueType::Int32 => self.read_i32().map(i64::from),
            ValueType::Bool => self.read_u8().map(i64::from),
            ValueType::Uint64 => self.read_u64().and_then(|v| i64::try_from(v).ok()),
            ValueType::Int64 => self.read_i64(),
            ValueType::Float32 | ValueType::Float64 | ValueType::String | ValueType::Array => None,
        }
    }

    /// Advances past a value of `ty` without materializing it. Arrays are
    /// skipped element-wise since string elements have per-element lengths (the
    /// tokenizer vocab is a ~250k-string array, so this loop is the hot path --
    /// each element is just a length read + seek).
    fn skip_value(&mut self, ty: ValueType) -> bool {
        match ty {
            ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => self.skip(1),
            ValueType::Uint16 | ValueType::Int16 => self.skip(2),
            ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => self.skip(4),
            ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => self.skip(8),
            ValueType::String => {
                let Some(length) = self.read_u64() else {
                    return false;
                };
                if length >= self.data.len() as u64 {
                    return false;
                }
                self.skip(length as usize)
            }
            ValueType::Array => {
                let Some(raw_elem) = self.read_u32() else {
                    return false;
                };
                let Some(elem) = ValueType::from_raw(raw_elem) else {
                    return false;
                };
                let Some(count) = self.read_u64() else {
                    return false;
                };
                if count >= self.data.len() as u64 {
                    return false;
                }
                // Fixed-size elements skip in one hop; strings/nested arrays iterate.
                match elem {
                    ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => {
                        self.skip(count as usize)
                    }
                    ValueType::Uint16 | ValueType::Int16 => self.skip((count as usize) * 2),
                    ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => {
                        self.skip((count as usize) * 4)
                    }
                    ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => {
                        self.skip((count as usize) * 8)
                    }
                    ValueType::String | ValueType::Array => {
                        for _ in 0..count {
                            if !self.skip_value(elem) {
                                return false;
                            }
                        }
                        true
                    }
                }
            }
        }
    }
}

/// Scans the KV section for the first key ending in `key_suffix`. The `Scan`
/// result distinguishes the two nil-ish outcomes: `Unparseable` means the header
/// couldn't be parsed at all; `Absent` means the header parsed cleanly but the
/// key isn't there (or isn't numeric) -- a trustworthy absence.
fn scan<V>(
    key_suffix: &str,
    path: &str,
    mut read: impl FnMut(&mut Cursor, ValueType) -> V,
) -> Scan<V> {
    let Ok(file) = File::open(path) else {
        return Scan::Unparseable;
    };
    // `.alwaysMapped` avoids reading the (potentially huge) file into memory;
    // parsing only faults in the header pages. An empty file (or any mapping
    // failure) is treated exactly like Swift's unreadable `Data` -> unparseable.
    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
        return Scan::Unparseable;
    };
    let mut cursor = Cursor::new(&mmap);

    // Magic "GGUF" (0x46554747 little-endian) + version. Versions 2 and 3 share
    // the layout we parse (v1 used 32-bit counts and predates every model we'd
    // encounter; anything newer than 3 may have changed the format, so fail soft
    // rather than guess).
    let Some(magic) = cursor.read_u32() else {
        return Scan::Unparseable;
    };
    if magic != 0x4655_4747 {
        return Scan::Unparseable;
    }
    let Some(version) = cursor.read_u32() else {
        return Scan::Unparseable;
    };
    if version != 2 && version != 3 {
        return Scan::Unparseable;
    }
    if cursor.read_u64().is_none() {
        // tensor count (unused)
        return Scan::Unparseable;
    }
    let Some(kv_count) = cursor.read_u64() else {
        return Scan::Unparseable;
    };
    if kv_count >= 1_000_000 {
        return Scan::Unparseable;
    }

    for _ in 0..kv_count {
        let Some(key) = cursor.read_string() else {
            return Scan::Unparseable;
        };
        let Some(raw_type) = cursor.read_u32() else {
            return Scan::Unparseable;
        };
        let Some(ty) = ValueType::from_raw(raw_type) else {
            return Scan::Unparseable;
        };

        if key.ends_with(key_suffix) {
            return Scan::Present(read(&mut cursor, ty));
        }
        if !cursor.skip_value(ty) {
            return Scan::Unparseable;
        }
    }
    Scan::Absent
}

/// Namespace for the GGUF header lookups (mirrors the Swift `enum GGUFMetadata`).
pub struct GGUFMetadata;

impl GGUFMetadata {
    /// The model's native (training) context window, per its
    /// `<arch>.context_length` metadata key -- the value llama.cpp loads into
    /// `hparams.n_ctx_train`. Like the MTP head below, we match on the key suffix
    /// rather than resolving `general.architecture` first, since the key is
    /// arch-namespaced but unique within a file.
    ///
    /// The leading dot in the suffix is load-bearing: the sibling key
    /// `<arch>.rope.scaling.original_context_length` ends in `_context_length`
    /// (underscore, not dot), so anchoring on `.context_length` excludes it.
    ///
    /// The double optional keeps the two nil-ish outcomes apart, and callers care
    /// about the difference: `None` = header unparseable, fall back to a default;
    /// `Some(None)` = header parsed fine but the key is genuinely absent, which
    /// means llama.cpp can't run the file at all (a non-LLM GGUF). `Some(Some(n))`
    /// is the length.
    pub fn context_length(path: &str) -> Option<Option<i64>> {
        match scan(".context_length", path, |c, t| c.read_int_value(t)) {
            Scan::Unparseable => None,
            Scan::Absent => Some(None),
            Scan::Present(v) => Some(v),
        }
    }

    /// Whether the GGUF at `path` carries an embedded multi-token-prediction head,
    /// per its `<arch>.nextn_predict_layers` metadata key (the same key llama.cpp
    /// loads into `hparams.n_layer_nextn`). We match on the key suffix rather than
    /// resolving `general.architecture` first -- the key is namespaced per arch
    /// but unique within a file. Returns `None` when the header can't be parsed.
    pub fn has_embedded_mtp_head(path: &str) -> Option<bool> {
        // `None` = header unparseable (caller falls back to heuristics); a clean
        // absence is a definitive "no head".
        match scan(".nextn_predict_layers", path, |c, t| c.read_int_value(t)) {
            Scan::Unparseable => None,
            Scan::Present(layers) => Some(layers.unwrap_or(0) > 0),
            Scan::Absent => Some(false),
        }
    }

    /// Whether the GGUF at `path` is a decision model -- one that answers typed
    /// questions with a probability per option via llama-server's
    /// `/v1/systemone` endpoint, rather than generating text. llama.cpp keys this
    /// off `<arch>.decision.type`, so we do too: present means a decision model,
    /// whatever its value. Suffix-matched like the keys above. Returns `None` when
    /// the header can't be parsed.
    pub fn is_decision_model(path: &str) -> Option<bool> {
        match scan(".decision.type", path, |_, _| ()) {
            Scan::Unparseable => None,
            Scan::Present(()) => Some(true),
            Scan::Absent => Some(false),
        }
    }
}
