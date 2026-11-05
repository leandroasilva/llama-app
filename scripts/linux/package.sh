#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: package.sh <version> <arch> [out-dir]}"
ARCH="${2:?usage: package.sh <version> <arch> [out-dir]}"
OUT_DIR="${3:-${OUT_DIR:-build/linux}}"

VERSION="${VERSION#v}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$ROOT/llama-gtk/data"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

case "$ARCH" in
    x86_64)
        DEB_ARCH="amd64"
        ;;
    aarch64 | arm64)
        ARCH="aarch64"
        DEB_ARCH="arm64"
        ;;
    *)
        echo "unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

echo "Building llama-gtk $VERSION for $ARCH"
cargo build --release --manifest-path "$ROOT/llama-gtk/Cargo.toml"

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/llama-gtk/target}"
BIN="$TARGET_DIR/release/llama-gtk"
if [ ! -x "$BIN" ]; then
    echo "missing release binary: $BIN" >&2
    exit 1
fi

STAGE="$WORK_DIR/stage"
mkdir -p \
    "$STAGE/usr/bin" \
    "$STAGE/usr/share/applications" \
    "$STAGE/usr/share/icons/hicolor/512x512/apps" \
    "$STAGE/usr/share/doc/llama-gtk"

install -m 0755 "$BIN" "$STAGE/usr/bin/llama-gtk"
strip --strip-unneeded "$STAGE/usr/bin/llama-gtk"
install -m 0644 "$DATA_DIR/app.llama.LlamaGtk.desktop" \
    "$STAGE/usr/share/applications/app.llama.LlamaGtk.desktop"
install -m 0644 "$DATA_DIR/app.llama.LlamaGtk.png" \
    "$STAGE/usr/share/icons/hicolor/512x512/apps/app.llama.LlamaGtk.png"
install -m 0644 "$ROOT/LICENSE" "$STAGE/usr/share/doc/llama-gtk/LICENSE"

TAR_NAME="llama-gtk-${VERSION}-linux-${ARCH}"
TAR_ROOT="$WORK_DIR/$TAR_NAME"
mkdir -p "$TAR_ROOT"
install -m 0755 "$STAGE/usr/bin/llama-gtk" "$TAR_ROOT/llama-gtk"
install -m 0644 "$DATA_DIR/app.llama.LlamaGtk.desktop" "$TAR_ROOT/app.llama.LlamaGtk.desktop"
install -m 0644 "$DATA_DIR/app.llama.LlamaGtk.png" "$TAR_ROOT/app.llama.LlamaGtk.png"
install -m 0644 "$ROOT/LICENSE" "$TAR_ROOT/LICENSE"
tar -C "$WORK_DIR" -czf "$OUT_DIR/${TAR_NAME}.tar.gz" "$TAR_NAME"

DEB_ROOT="$WORK_DIR/deb"
mkdir -p "$DEB_ROOT/DEBIAN"
cp -a "$STAGE/." "$DEB_ROOT/"

SHLIB_DEPS=""
mkdir -p "$WORK_DIR/shlib/debian"
cat > "$WORK_DIR/shlib/debian/control" <<EOF
Source: llama-gtk
Section: utils
Priority: optional
Maintainer: Llama contributors <noreply@github.com>
Package: llama-gtk
Architecture: any
EOF
SHLIB_DEPS="$(cd "$WORK_DIR/shlib" && dpkg-shlibdeps -O "$DEB_ROOT/usr/bin/llama-gtk" 2>/dev/null | sed -n 's/^shlibs:Depends=//p' || true)"
if [ -z "$SHLIB_DEPS" ]; then
    SHLIB_DEPS="libc6 (>= 2.35), libgcc-s1, libgtk-4-1 (>= 4.6.0)"
fi

cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: llama-gtk
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${DEB_ARCH}
Maintainer: Llama contributors <noreply@github.com>
Depends: ${SHLIB_DEPS}, ca-certificates
Description: Llama desktop application for Linux
 A GTK4 frontend for llama.cpp that manages local models, downloads from
 Hugging Face, and runs an OpenAI-compatible server.
EOF

dpkg-deb --build --root-owner-group "$DEB_ROOT" \
    "$OUT_DIR/llama-gtk_${VERSION}_${DEB_ARCH}.deb"

RPM_ROOT="$WORK_DIR/rpm"
mkdir -p "$RPM_ROOT/BUILD" "$RPM_ROOT/BUILDROOT" "$RPM_ROOT/RPMS" \
    "$RPM_ROOT/SOURCES" "$RPM_ROOT/SPECS" "$RPM_ROOT/SRPMS"
SPEC="$RPM_ROOT/SPECS/llama-gtk.spec"

cat > "$SPEC" <<EOF
Name:           llama-gtk
Version:        ${VERSION}
Release:        1
Summary:        Llama desktop application for Linux
License:        MIT
URL:            https://github.com/leandroasilva/llama-app
AutoReqProv:    yes
Requires:       ca-certificates

%description
A GTK4 frontend for llama.cpp that manages local models, downloads from
Hugging Face, and runs an OpenAI-compatible server.

%prep

%build

%install
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/applications
mkdir -p %{buildroot}/usr/share/icons/hicolor/512x512/apps
mkdir -p %{buildroot}/usr/share/doc/llama-gtk
install -m 0755 "${STAGE}/usr/bin/llama-gtk" %{buildroot}/usr/bin/llama-gtk
install -m 0644 "${DATA_DIR}/app.llama.LlamaGtk.desktop" %{buildroot}/usr/share/applications/app.llama.LlamaGtk.desktop
install -m 0644 "${DATA_DIR}/app.llama.LlamaGtk.png" %{buildroot}/usr/share/icons/hicolor/512x512/apps/app.llama.LlamaGtk.png
install -m 0644 "${ROOT}/LICENSE" %{buildroot}/usr/share/doc/llama-gtk/LICENSE

%files
/usr/bin/llama-gtk
/usr/share/applications/app.llama.LlamaGtk.desktop
/usr/share/icons/hicolor/512x512/apps/app.llama.LlamaGtk.png
%license /usr/share/doc/llama-gtk/LICENSE

%changelog
* $(date '+%a %b %d %Y') Llama contributors <noreply@github.com> - ${VERSION}-1
- Release ${VERSION}
EOF

rpmbuild -bb --target "$ARCH" --define "_topdir $RPM_ROOT" \
    --define "debug_package %{nil}" "$SPEC"
find "$RPM_ROOT/RPMS" -name '*.rpm' -exec cp {} "$OUT_DIR/" \;

(
    cd "$OUT_DIR"
    sha256sum "llama-gtk-${VERSION}-linux-${ARCH}.tar.gz" \
        "llama-gtk_${VERSION}_${DEB_ARCH}.deb" \
        "llama-gtk-${VERSION}-1.${ARCH}.rpm" \
        > "llama-gtk-${VERSION}-linux-${ARCH}.sha256"
)

echo "Linux packages written to $OUT_DIR:"
ls -lh "$OUT_DIR" | sed -n '/llama-gtk/p'
