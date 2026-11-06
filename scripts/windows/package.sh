#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: package.sh <version> [out-dir]}"
OUT_DIR="${2:-${OUT_DIR:-build/windows}}"

VERSION="${VERSION#v}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$ROOT/llama-gtk/data"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

TARGET="x86_64-pc-windows-gnu"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/llama-gtk/target}"
BIN="$TARGET_DIR/$TARGET/release/llama-gtk.exe"
MINGW_PREFIX="${MSYSTEM_PREFIX:-/mingw64}"
NAME="llama-gtk-${VERSION}-windows-x86_64"
BUNDLE="$WORK_DIR/$NAME"

if [ ! -d "$MINGW_PREFIX" ]; then
    echo "missing MSYS2 mingw prefix: $MINGW_PREFIX" >&2
    exit 1
fi

echo "Building llama-gtk $VERSION for windows-x86_64"
cargo build --locked --release --target "$TARGET" --manifest-path "$ROOT/llama-gtk/Cargo.toml"
if [ ! -f "$BIN" ]; then
    echo "missing release binary: $BIN" >&2
    exit 1
fi

mkdir -p \
    "$BUNDLE/lib/gdk-pixbuf-2.0/2.10.0/loaders" \
    "$BUNDLE/share"

install -m 0755 "$BIN" "$BUNDLE/llama-gtk.exe"
if command -v strip >/dev/null 2>&1; then
    strip --strip-unneeded "$BUNDLE/llama-gtk.exe" || true
fi

install -m 0644 "$DATA_DIR/app.llama.LlamaGtk.png" "$BUNDLE/app.llama.LlamaGtk.png"
install -m 0644 "$ROOT/LICENSE" "$BUNDLE/LICENSE"

declare -A COPIED_DLLS=()
copy_dlls() {
    local queue=("$1")
    local index=0
    while [ "$index" -lt "${#queue[@]}" ]; do
        local binary="${queue[$index]}"
        index=$((index + 1))
        while IFS= read -r dll; do
            [ -f "$dll" ] || continue
            case "$dll" in
                "$MINGW_PREFIX"/* | /mingw64/* | *"/mingw64/"*) ;;
                *) continue ;;
            esac
            local base
            base="$(basename "$dll")"
            if [ -z "${COPIED_DLLS[$base]:-}" ]; then
                COPIED_DLLS[$base]=1
                cp -f "$dll" "$BUNDLE/"
                queue+=("$dll")
            fi
        done < <(ldd "$binary" 2>/dev/null | awk '/=>/ {print $3}')
    done
}

copy_dlls "$BUNDLE/llama-gtk.exe"

for dir in glib-2.0 icons gtk-4.0 mime; do
    if [ -d "$MINGW_PREFIX/share/$dir" ]; then
        mkdir -p "$BUNDLE/share/$dir"
        cp -a "$MINGW_PREFIX/share/$dir/." "$BUNDLE/share/$dir/"
    fi
done

if [ -d "$MINGW_PREFIX/lib/gtk-4.0" ]; then
    mkdir -p "$BUNDLE/lib/gtk-4.0"
    cp -a "$MINGW_PREFIX/lib/gtk-4.0/." "$BUNDLE/lib/gtk-4.0/"
fi

if [ -d "$MINGW_PREFIX/lib/gio/modules" ]; then
    mkdir -p "$BUNDLE/lib/gio/modules"
    cp -a "$MINGW_PREFIX/lib/gio/modules/." "$BUNDLE/lib/gio/modules/"
fi

if [ -d "$MINGW_PREFIX/lib/gdk-pixbuf-2.0/2.10.0/loaders" ]; then
    cp -a "$MINGW_PREFIX/lib/gdk-pixbuf-2.0/2.10.0/loaders/." \
        "$BUNDLE/lib/gdk-pixbuf-2.0/2.10.0/loaders/"
fi

while IFS= read -r plugin; do
    copy_dlls "$plugin"
done < <(find "$BUNDLE/lib" -type f -name '*.dll' 2>/dev/null)

if [ -d "$BUNDLE/share/glib-2.0/schemas" ] && command -v glib-compile-schemas >/dev/null 2>&1; then
    glib-compile-schemas "$BUNDLE/share/glib-2.0/schemas"
fi

if ls "$BUNDLE/lib/gdk-pixbuf-2.0/2.10.0/loaders/"*.dll >/dev/null 2>&1 \
    && command -v gdk-pixbuf-query-loaders >/dev/null 2>&1; then
    (
        cd "$BUNDLE"
        GDK_PIXBUF_MODULEDIR=lib/gdk-pixbuf-2.0/2.10.0/loaders \
            gdk-pixbuf-query-loaders lib/gdk-pixbuf-2.0/2.10.0/loaders/*.dll \
            > lib/gdk-pixbuf-2.0/2.10.0/loaders.cache
    )
    sed -i -E \
        's#"[^"]*[\\/]+lib[\\/]+gdk-pixbuf-2\.0[\\/]+2\.10\.0[\\/]+loaders[\\/]+([^"\\/]+)"#"lib/gdk-pixbuf-2.0/2.10.0/loaders/\1"#g' \
        "$BUNDLE/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"
fi

(
    cd "$WORK_DIR"
    zip -qr "$OUT_DIR/$NAME.zip" "$NAME"
)

(
    cd "$OUT_DIR"
    sha256sum "$NAME.zip" > "$NAME.sha256"
)

echo "Windows package written to $OUT_DIR:"
ls -lh "$OUT_DIR/$NAME.zip" "$OUT_DIR/$NAME.sha256"
