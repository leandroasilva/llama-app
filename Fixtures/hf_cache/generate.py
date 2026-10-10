#!/usr/bin/env python3
"""Generates the HF-cache golden tree and its expectations.

Run from anywhere: `python3 Fixtures/hf_cache/generate.py`. It writes
`Fixtures/hf_cache/cache/` (a read-only corpus; the tests copy it before any
mutating operation) and `Fixtures/hf_cache.json`, both consumed by
`llama-core/tests/phase2.rs` and `LlamaTests/Phase2FixtureTests.swift`.
"""

import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_DIR = os.path.join(HERE, "..", "gguf")
CACHE = os.path.join(HERE, "cache")
OUT_JSON = os.path.join(HERE, "..", "hf_cache.json")


def blob(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_gguf(name: str) -> bytes:
    with open(os.path.join(GGUF_DIR, name), "rb") as fh:
        return fh.read()


FULL = read_gguf("full.gguf")          # ctx 40960, embedded MTP
NOMTP = read_gguf("nomtp.gguf")        # ctx 8192, no head
DECISION = read_gguf("decision.gguf")  # ctx 32768, decision model

# Sidecar payloads. The scan never parses sidecar metadata, so these are opaque
# bytes -- their only job is to pin the byte size the entry sums in.
MMPROJ = b"mmproj-sidecar"
MTP_SIDECAR = b"mtp-sidecar"

# id -> content, so sizes and blob names are deterministic.
CONTENT = {
    "full": FULL,
    "nomtp": NOMTP,
    "decision": DECISION,
}


def write_blob(repo: str, content: bytes) -> str:
    h = blob(content)
    d = os.path.join(CACHE, repo, "blobs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, h), "wb") as fh:
        fh.write(content)
    return h


def link(repo: str, commit: str, filename: str, h: str):
    link_path = os.path.join(CACHE, repo, "snapshots", commit, filename)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    target = os.path.relpath(os.path.join(CACHE, repo, "blobs", h), os.path.dirname(link_path))
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(target, link_path)


def write_ref(repo: str, commit: str):
    d = os.path.join(CACHE, repo, "refs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "main"), "w") as fh:
        fh.write(commit)


if os.path.exists(CACHE):
    shutil.rmtree(CACHE)

# 1. Tiny-GGUF: a main GGUF + an mmproj sidecar.
h_full = write_blob("models--unsloth--Tiny-GGUF", NOMTP)
h_mmproj = write_blob("models--unsloth--Tiny-GGUF", MMPROJ)
link("models--unsloth--Tiny-GGUF", "commit1", "tiny-Q4_K_M.gguf", h_full)
link("models--unsloth--Tiny-GGUF", "commit1", "mmproj-f16.gguf", h_mmproj)
write_ref("models--unsloth--Tiny-GGUF", "commit1")

# 2. Sharded-GGUF: a complete 2-shard group in a quant subdir.
h_s1 = write_blob("models--unsloth--Sharded-GGUF", NOMTP)
h_s2 = write_blob("models--unsloth--Sharded-GGUF", FULL)
link("models--unsloth--Sharded-GGUF", "commit2", "Q4_K_M/shard-Q4_K_M-00001-of-00002.gguf", h_s1)
link("models--unsloth--Sharded-GGUF", "commit2", "Q4_K_M/shard-Q4_K_M-00002-of-00002.gguf", h_s2)
write_ref("models--unsloth--Sharded-GGUF", "commit2")

# 3. MTP-GGUF: main + a draft-head sidecar (no embedded head).
h_main = write_blob("models--unsloth--MTP-GGUF", NOMTP)
h_mtp = write_blob("models--unsloth--MTP-GGUF", MTP_SIDECAR)
link("models--unsloth--MTP-GGUF", "commit3", "model-Q8_0.gguf", h_main)
link("models--unsloth--MTP-GGUF", "commit3", "mtp-Q8_0.gguf", h_mtp)
write_ref("models--unsloth--MTP-GGUF", "commit3")

# 4. Decision-GGUF: a decision model.
h_dec = write_blob("models--org--Decision-GGUF", DECISION)
link("models--org--Decision-GGUF", "commit4", "model-Q4_K_M.gguf", h_dec)
write_ref("models--org--Decision-GGUF", "commit4")

# 5. Embedded-GGUF: embedded MTP head, no sidecar.
h_emb = write_blob("models--org--Embedded-GGUF", FULL)
link("models--org--Embedded-GGUF", "commit5", "embed-Q4_K_M.gguf", h_emb)
write_ref("models--org--Embedded-GGUF", "commit5")

# 6. Partial-GGUF: an INCOMPLETE shard group (2 of 3) -> must be skipped.
h_p1 = write_blob("models--org--Partial-GGUF", FULL)
h_p2 = write_blob("models--org--Partial-GGUF", NOMTP)
link("models--org--Partial-GGUF", "commit6", "inc-Q4_K_M-00001-of-00003.gguf", h_p1)
link("models--org--Partial-GGUF", "commit6", "inc-Q4_K_M-00002-of-00003.gguf", h_p2)
write_ref("models--org--Partial-GGUF", "commit6")

# 7. Staging partials for Tiny-GGUF + an unrelated org.
pdir = os.path.join(CACHE, ".llama-partial", "unsloth", "Tiny-GGUF:Q4_K_M")
os.makedirs(pdir, exist_ok=True)
with open(os.path.join(pdir, ".model.json"), "w") as fh:
    fh.write('{"id":"unsloth/Tiny-GGUF:Q4_K_M"}')
with open(os.path.join(pdir, "tiny-Q4_K_M.gguf.partial"), "wb") as fh:
    fh.write(b"0123456789ab")  # 12 bytes
with open(os.path.join(pdir, "extra.partial"), "wb") as fh:
    fh.write(b"01234567")  # 8 bytes

odir = os.path.join(CACHE, ".llama-partial", "org", "Other-GGUF:Q8_0")
os.makedirs(odir, exist_ok=True)
with open(os.path.join(odir, ".model.json"), "w") as fh:
    fh.write('{"id":"org/Other-GGUF:Q8_0"}')

# ---- Expectations ----

entries = [
    {
        "id": "org/Decision-GGUF:Q4_K_M",
        "ctxWindow": 32768,
        "fileSize": len(DECISION),
        "draftHead": None,
        "isDecisionModel": True,
        "usesMTP": False,
        "mmprojFile": None,
        "draftSidecarFile": None,
        "additionalParts": [],
        "hfRepoDirName": "models--org--Decision-GGUF",
        "modelFile": "models--org--Decision-GGUF/snapshots/commit4/model-Q4_K_M.gguf",
    },
    {
        "id": "org/Embedded-GGUF:Q4_K_M",
        "ctxWindow": 40960,
        "fileSize": len(FULL),
        "draftHead": "mtp",
        "isDecisionModel": False,
        "usesMTP": True,
        "mmprojFile": None,
        "draftSidecarFile": None,
        "additionalParts": [],
        "hfRepoDirName": "models--org--Embedded-GGUF",
        "modelFile": "models--org--Embedded-GGUF/snapshots/commit5/embed-Q4_K_M.gguf",
    },
    {
        "id": "unsloth/MTP-GGUF:Q8_0",
        "ctxWindow": 8192,
        "fileSize": len(NOMTP),
        "draftHead": "mtp",
        "isDecisionModel": False,
        "usesMTP": False,
        "mmprojFile": None,
        "draftSidecarFile": "models--unsloth--MTP-GGUF/snapshots/commit3/mtp-Q8_0.gguf",
        "additionalParts": [],
        "hfRepoDirName": "models--unsloth--MTP-GGUF",
        "modelFile": "models--unsloth--MTP-GGUF/snapshots/commit3/model-Q8_0.gguf",
    },
    {
        "id": "unsloth/Sharded-GGUF:Q4_K_M",
        "ctxWindow": 8192,
        "fileSize": len(NOMTP) + len(FULL),
        "draftHead": None,
        "isDecisionModel": False,
        "usesMTP": False,
        "mmprojFile": None,
        "draftSidecarFile": None,
        "additionalParts": [
            "models--unsloth--Sharded-GGUF/snapshots/commit2/Q4_K_M/shard-Q4_K_M-00002-of-00002.gguf"
        ],
        "hfRepoDirName": "models--unsloth--Sharded-GGUF",
        "modelFile": "models--unsloth--Sharded-GGUF/snapshots/commit2/Q4_K_M/shard-Q4_K_M-00001-of-00002.gguf",
    },
    {
        "id": "unsloth/Tiny-GGUF:Q4_K_M",
        "ctxWindow": 8192,
        "fileSize": len(NOMTP) + len(MMPROJ),
        "draftHead": None,
        "isDecisionModel": False,
        "usesMTP": False,
        "mmprojFile": "models--unsloth--Tiny-GGUF/snapshots/commit1/mmproj-f16.gguf",
        "draftSidecarFile": None,
        "additionalParts": [],
        "hfRepoDirName": "models--unsloth--Tiny-GGUF",
        "modelFile": "models--unsloth--Tiny-GGUF/snapshots/commit1/tiny-Q4_K_M.gguf",
    },
]

fixture = {
    "pathHelpers": {
        "repoDirName": [
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/file.gguf",
                "name": "models--unsloth--Qwen3.5-2B-GGUF",
            },
            {
                "url": "https://huggingface.co/org/repo/resolve/main/Q4_K_M/model.gguf",
                "name": "models--org--repo",
            },
            {"url": "https://huggingface.co/org/repo/blob/main/file.gguf", "name": None},
            {"url": "https://huggingface.co/org/repo", "name": None},
        ],
        "repoRelativePath": [
            {
                "url": "https://huggingface.co/org/repo/resolve/main/Q4_K_M/model.gguf",
                "path": "Q4_K_M/model.gguf",
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/file.gguf",
                "path": "file.gguf",
            },
            {"url": "https://huggingface.co/org/repo/resolve/main", "path": None},
        ],
    },
    "scan": {"cacheDir": "hf_cache/cache", "entries": entries},
    "partialBytes": [
        {"modelId": "unsloth/Tiny-GGUF:Q4_K_M", "bytes": 20},
        {"modelId": "org/Other-GGUF:Q8_0", "bytes": 0},
        {"modelId": "unsloth/Missing-GGUF:Q4_K_M", "bytes": 0},
    ],
    "scanPlaceholders": {
        "cacheDir": "hf_cache/cache",
        "paths": [
            ".llama-partial/unsloth/Tiny-GGUF:Q4_K_M/.model.json",
            ".llama-partial/org/Other-GGUF:Q8_0/.model.json",
        ],
    },
    "snapshotFileExists": [
        {"repoDir": "models--unsloth--Tiny-GGUF", "filename": "tiny-Q4_K_M.gguf", "value": True},
        {"repoDir": "models--unsloth--Tiny-GGUF", "filename": "nope.gguf", "value": False},
        {"repoDir": "models--unsloth--Tiny-GGUF", "filename": "Q4_K_M/model.gguf", "value": False},
        {"repoDir": "models--nope--Missing", "filename": "x.gguf", "value": False},
    ],
    # Consumed by the mutating tests in each language after copying the tree.
    "writeBlobAndLink": {
        "repoDir": "models--test--wbl",
        "commit": "deadbeefcafe",
        "blobHash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "filename": "Q4_K_M/model-Q4_K_M.gguf",
        "expectedSymlinkTarget": "../../../blobs/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "expectedRefsMain": "deadbeefcafe",
    },
    "deleteModelFiles": {
        "repoDir": "models--unsloth--Tiny-GGUF",
        "modelFile": "models--unsloth--Tiny-GGUF/snapshots/commit1/tiny-Q4_K_M.gguf",
        "additionalParts": [],
        "mmprojFile": "models--unsloth--Tiny-GGUF/snapshots/commit1/mmproj-f16.gguf",
        "expected": {
            "modelFileExists": False,
            "mmprojFileExists": False,
            "blobsDirExists": False,
            "snapshotsDirExists": False,
            "repoDirExists": True,
            "refsMainExists": True,
        },
    },
}

with open(OUT_JSON, "w") as fh:
    json.dump(fixture, fh, indent=2)
    fh.write("\n")

print(f"wrote tree + {os.path.relpath(OUT_JSON, HERE)} ({len(entries)} scan entries)")
