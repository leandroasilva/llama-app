# Llama

Llama is a macOS menu bar app for running local LLMs.

[Watch a 2-minute intro](https://www.youtube.com/watch?v=7AieF7rZUTc)

<br>

![Llama](https://github.com/user-attachments/assets/df78f9ee-bb1d-4883-bf08-44371b0cd58a)

<br>

## Install

```sh
brew install --cask llama-app
```

Or download from [Releases](https://github.com/ggml-org/Llama-macOS/releases).

## How it works

When you start Llama, it runs a local server at `http://localhost:9931/v1`.

If you have llama.cpp installed, Llama uses it. Otherwise, it installs a prebuilt binary for your Mac. Models you've already installed via llama.cpp show up in the app automatically. You can install any GGUF model from Hugging Face, and Llama also recommends models that fit your Mac's hardware.

You can chat with any model in the built-in WebUI, connect other apps (coding agents, chat UIs, editors), or use the API directly. Models load when requested and unload when idle, so they don't take up memory when not in use.

## Features

### Core

- **100% local** -- models run on your Mac; no data ever leaves it
- **Small footprint** -- 4 MB native macOS app (AppKit + SwiftUI)
- **Zero configuration** -- models are auto-configured with optimal settings for your Mac's hardware
- **Router mode** -- the server process runs continuously; models are loaded and unloaded in-place without restarting
- **Built on llama.cpp** -- from the GGML org, developed alongside llama.cpp

### Model management

- **One-click install** -- browse recommended models or install any GGUF from Hugging Face
- **Hardware-aware recommendations** -- the catalog filters models by your Mac's memory and selects the best quantization automatically
- **Memory profiling** -- each model is profiled at install time to determine which context lengths (4k--256k) fit in your device memory
- **Pause and resume** -- downloads survive network drops and app restarts, with automatic retry and exponential backoff
- **Sharded models** -- full support for multi-file GGUF sharded models, vision sidecars (mmproj), and speculative decoding draft heads
- **Standard storage** -- models live in the Hugging Face cache (`~/.cache/huggingface/hub`), shared with llama.cpp and other tools; the directory is configurable
- **Model aliases** -- assign short names to models (e.g. `code` -> `ggml-org/some-model:Q4_K_M`) for convenience in API calls
- **Delete and reinstall** -- manage models directly from the menu with confirmation guards

### Context and memory

- **Context tier picker** -- choose context length per model (4k, 8k, 16k, 32k, 64k, 128k, 256k) with projected memory cost shown for each tier
- **Tiers that exceed device memory are disabled** -- the app prevents you from selecting a context that won't fit
- **Idle unloading** -- configure how long the server waits before unloading an idle model (5m, 15m, 1h, or off)

### Server and API

- **OpenAI-compatible API** -- standard `/v1/chat/completions`, `/v1/models`, and all llama.cpp server endpoints
- **Configurable port** -- change the server port from the default 9931
- **In-place model reload** -- the model list is refreshed without restarting the server when you install or delete models
- **Server log viewer** -- open the llama-server log directly in Console.app from Settings

### Network access

- **Localhost only by default** -- the server is reachable only from your Mac
- **Tailscale** -- bind to your Tailscale address for secure remote access from your other devices, authenticated and encrypted by Tailscale
- **This network** -- bind to all interfaces (`0.0.0.0`) so anything on your local network can reach the server
- **Network change detection** -- the app monitors network path changes and withdraws access when the Mac joins a different network
- **QR code** -- scan the server URL QR code from the menu bar to connect from a phone or other device

### Global input

- **System-wide hotkey** -- configure a keyboard shortcut in Settings to open a Spotlight-style prompt from anywhere
- **Floating capture panel** -- type a prompt without switching apps; the panel appears over your current window
- **Model switcher** -- press `Cmd+K` in the panel to pick a different model before sending
- **Smart default** -- the panel targets the last used model, or the currently active model, or the smallest installed model

### Chat and WebUI

- **Built-in WebUI** -- chat with any model in the browser via the "Open chat" link
- **Agent mode** -- toggle in Settings to allow the WebUI to make API requests with elevated permissions
- **Custom web app** -- point the chat UI to your own web app directory (must contain `index.html`)
- **Quick prompt shortcut** -- assign a keyboard shortcut in Settings to open the WebUI with a pre-filled prompt
- **Request builder** -- generate API requests for any installed model from its detail page, with the correct base URL and model ID pre-filled

### Deeplinks

- **URL scheme** -- `llama://install?repo=org/repo&quant=Q4_K_M` triggers model installation from anywhere (browser, terminal, other apps)
- **Smart file selection** -- when no quantization is specified, the app picks Q4_K_M, falls back to Q4_0, then to the largest file that fits in device memory
- **Automatic sidecar download** -- installs mmproj (vision encoder) and draft head (speculative decoding) files alongside the main model

### Custom model settings

The app configures each model for your Mac and writes that config to `models.ini`, which it regenerates on every launch. To change a setting, or add one the app doesn't set, edit `~/.config/llama/models.user.ini` instead -- the app only reads that file, and merges it into the config it generates.

Use the same section header the app uses (`org/repo:QUANT`), and list only the keys you want to change. Keys are `llama serve` options without the leading dashes.

```ini
[ggml-org/gemma-4-E4B-it-GGUF:Q8_0]
temp = 0.7
ctx-size = 32768
cache-type-k = q4_1
```

A section for a model the app didn't find is passed through as-is, which is how you point at weights or a draft model from another repo.

```ini
[unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-Q2_K_XL]
model = /path/to/DeepSeek-V4-Flash-UD-Q2_K_XL.gguf
spec-type = draft-dspark
spec-draft-model = /path/to/draft.gguf
```

Anything you set here wins over the app's own value, including settings derived from how much memory your Mac has. If a key can't be applied, the app ignores the file and the menu says which option was at fault.

A commented template is written to this path on first launch.

### System integration

- **Launch at login** -- enabled by default on first launch, toggleable in Settings
- **Auto-update** -- Sparkle framework checks for updates automatically; click the version number in the menu to check manually
- **AppleScript support** -- `show menu`, `show settings`, `show settings tab "Name"`, `show global input`
- **Menu bar only** -- no Dock icon (`LSUIElement`); the app lives entirely in the status bar

## Example requests

List installed models:

```sh
curl http://localhost:9931/v1/models
```

Send a message to a model:

```sh
curl http://localhost:9931/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ggml-org/gpt-oss-20b-GGUF:MXFP4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

See complete API reference in the llama.cpp server [docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#api-endpoints).

## Advanced settings

**Bind to a specific address** -- to pin the server to one address the app doesn't offer, set it by hand. The app leaves it alone, and falls back to localhost whenever the address isn't on any interface.

```sh
defaults write app.llama.Llama exposeToNetwork -string "192.168.1.50"
```

**Custom server arguments** -- extra CLI arguments appended to the `llama serve` command, for server flags the app doesn't expose (e.g. `--api-key`). They come after the app's own flags, so where the server honors the later occurrence they can override the app's settings. Takes effect on the next server start.

```sh
# append custom arguments to the server command
defaults write app.llama.Llama extraServerArgs -string "--api-key secret"

# remove (default)
defaults delete app.llama.Llama extraServerArgs
```

**Hugging Face token** -- set a `hf_` token in Settings > Advanced to access gated repositories.

## Requirements

- macOS 15 (Sequoia) or later
- Apple Silicon (arm64)
