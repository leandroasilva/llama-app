.PHONY: build build-release test test-rust test-swift lint lint-rust lint-swift fmt fmt-rust fmt-swift clean

# Default target
all: build

# --- Build ---

build: build-release

build-release:
	@echo "Building Rust core (release)..."
	cargo build --release -p llama-core-ffi
	@echo "Building Swift app..."
	xcodebuild build \
		-scheme Llama \
		-configuration Release \
		-derivedDataPath build \
		ARCHS=arm64 \
		CODE_SIGN_IDENTITY=- \
		CODE_SIGNING_REQUIRED=NO \
		CODE_SIGNING_ALLOWED=NO

build-debug:
	@echo "Building Rust core (debug)..."
	cargo build -p llama-core-ffi
	@echo "Building Swift app (debug)..."
	xcodebuild build \
		-scheme Llama \
		-configuration Debug \
		-derivedDataPath build \
		ARCHS=arm64 \
		CODE_SIGN_IDENTITY=- \
		CODE_SIGNING_REQUIRED=NO \
		CODE_SIGNING_ALLOWED=NO

# --- Test ---

test: test-rust test-swift

test-rust:
	@echo "Running Rust tests..."
	cargo test --all

test-swift:
	@echo "Running Swift tests..."
	xcodebuild test \
		-scheme Llama \
		-configuration Debug \
		-derivedDataPath build \
		ARCHS=arm64 \
		CODE_SIGN_IDENTITY=- \
		CODE_SIGNING_REQUIRED=NO \
		CODE_SIGNING_ALLOWED=NO

# --- Lint ---

lint: lint-rust lint-swift

lint-rust:
	@echo "Running cargo fmt check..."
	cargo fmt --all -- --check
	@echo "Running cargo clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

lint-swift:
	@echo "Swift lint not configured — add SwiftLint to enable."
	@exit 0

# --- Format ---

fmt: fmt-rust fmt-swift

fmt-rust:
	@echo "Formatting Rust code..."
	cargo fmt --all

fmt-swift:
	@echo "Swift format not configured — add .swiftformat to enable."
	@exit 0

# --- Clean ---

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf build

# --- Help ---

help:
	@echo "Available targets:"
	@echo "  build          Build Rust core (release) + Swift app"
	@echo "  build-debug    Build Rust core (debug) + Swift app"
	@echo "  test           Run all tests (Rust + Swift)"
	@echo "  test-rust      Run Rust tests only"
	@echo "  test-swift     Run Swift tests only"
	@echo "  lint           Run all linters (cargo fmt --check, cargo clippy)"
	@echo "  lint-rust      Run Rust linters only"
	@echo "  fmt            Format all code"
	@echo "  fmt-rust       Format Rust code only"
	@echo "  clean          Remove build artifacts"
