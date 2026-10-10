#!/usr/bin/env python3
"""Generates the synthetic GGUF golden files and their expectations.

Run from anywhere: `python3 Fixtures/gguf/generate.py`. The `.gguf` binaries and
`Fixtures/gguf_metadata.json` it writes are the shared corpus consumed by both
`llama-core/tests/phase2.rs` and `LlamaTests/Phase2FixtureTests.swift`; regenerate
only when the cases change, and commit the outputs.
"""

import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "gguf_metadata.json")

# On-disk value type ids (gguf_metadata_value_type).
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)


def s(value: str) -> bytes:
    b = value.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def kv(key: str, type_id: int, payload: bytes) -> bytes:
    return s(key) + u32(type_id) + payload


def arr_str(items):
    body = u32(STR) + u64(len(items))
    for it in items:
        body += s(it)
    return body


def gguf(version: int, kvs: list) -> bytes:
    out = b"GGUF" + u32(version) + u64(0) + u64(len(kvs))
    for k in kvs:
        out += k
    return out


CASES = []


def case(name, version, kvs, ctx, mtp, decision):
    CASES.append(
        {
            "file": f"gguf/{name}",
            "contextLength": ctx,
            "hasEmbeddedMTPHead": mtp,
            "isDecisionModel": decision,
        }
    )
    with open(os.path.join(HERE, name), "wb") as fh:
        fh.write(gguf(version, kvs))


# Both a decoy key ending in `_context_length` (which must not match the
# `.context_length` suffix) and a vector array (exercised by the array skip).
case(
    "full.gguf",
    3,
    [
        kv("general.architecture", STR, s("qwen35moe")),
        kv("tokenizer.ggml.tokens", ARR, arr_str(["a", "bb", "ccc"])),
        kv("qwen35moe.rope.scaling.original_context_length", U32, u32(4096)),
        kv("qwen35moe.context_length", U32, u32(40960)),
        kv("qwen35moe.nextn_predict_layers", U32, u32(1)),
    ],
    ctx={"parsed": True, "length": 40960},
    mtp={"parsed": True, "value": True},
    decision={"parsed": True, "value": False},
)

case(
    "nomtp.gguf",
    3,
    [
        kv("general.architecture", STR, s("gemma3")),
        kv("gemma3.context_length", U32, u32(8192)),
    ],
    ctx={"parsed": True, "length": 8192},
    mtp={"parsed": True, "value": False},
    decision={"parsed": True, "value": False},
)

# uint64 context length exercises the wide-int path; decision.type is present.
case(
    "decision.gguf",
    2,
    [
        kv("general.architecture", STR, s("qwen3")),
        kv("qwen3.context_length", U64, u64(32768)),
        kv("qwen3.decision.type", STR, s("classification")),
    ],
    ctx={"parsed": True, "length": 32768},
    mtp={"parsed": True, "value": False},
    decision={"parsed": True, "value": True},
)

# Header parses cleanly but the context key is genuinely absent -> Some(None).
case(
    "nocontext.gguf",
    3,
    [
        kv("general.architecture", STR, s("cohere_asr")),
        kv("stt.something", U32, u32(5)),
    ],
    ctx={"parsed": True, "length": None},
    mtp={"parsed": True, "value": False},
    decision={"parsed": True, "value": False},
)

# nextn_predict_layers == 0 is a definitive "no head".
case(
    "zeromtp.gguf",
    3,
    [
        kv("general.architecture", STR, s("llama")),
        kv("llama.context_length", U32, u32(131072)),
        kv("llama.nextn_predict_layers", U32, u32(0)),
    ],
    ctx={"parsed": True, "length": 131072},
    mtp={"parsed": True, "value": False},
    decision={"parsed": True, "value": False},
)

# Unparseable cases: each must yield `parsed: false` for every lookup.

# Invalid magic.
with open(os.path.join(HERE, "notgguf.gguf"), "wb") as fh:
    fh.write(b"JUNKJUNK" * 4)
CASES.append(
    {
        "file": "gguf/notgguf.gguf",
        "contextLength": {"parsed": False},
        "hasEmbeddedMTPHead": {"parsed": False},
        "isDecisionModel": {"parsed": False},
    }
)

# Unsupported (v1) version.
case(
    "v1.gguf",
    1,
    [kv("general.architecture", STR, s("llama"))],
    ctx={"parsed": False},
    mtp={"parsed": False},
    decision={"parsed": False},
)

# Truncated header: declares 10 KV pairs but carries none.
with open(os.path.join(HERE, "truncated.gguf"), "wb") as fh:
    fh.write(b"GGUF" + u32(3) + u64(0) + u64(10))
CASES.append(
    {
        "file": "gguf/truncated.gguf",
        "contextLength": {"parsed": False},
        "hasEmbeddedMTPHead": {"parsed": False},
        "isDecisionModel": {"parsed": False},
    }
)

# Unknown value type id (99) in the first pair -> fail soft.
case(
    "unknowntype.gguf",
    3,
    [kv("general.architecture", 99, b"")],
    ctx={"parsed": False},
    mtp={"parsed": False},
    decision={"parsed": False},
)

with open(OUT_JSON, "w") as fh:
    json.dump({"files": CASES}, fh, indent=2)
    fh.write("\n")

print(f"wrote {len(CASES)} cases + {os.path.relpath(OUT_JSON, HERE)}")
