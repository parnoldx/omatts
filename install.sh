#!/bin/sh
# omatts installer
#   binary  -> ~/.local/bin/omatts
#   models  -> ~/.local/share/omatts/models
#   voices  -> ~/.local/share/omatts/voices
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/parnoldx/omatts/master/install.sh | sh
#
# Optional overrides (env vars):
#   OMATTS_REPO=paarnold/omatts      GitHub repo (must match this script's home)
#   OMATTS_VERSION=latest            release tag, or e.g. v2026.09.11
#   OMATTS_URL=...                   direct tarball URL, skips repo/version
#   PREFIX=/usr/local                install root (default ~/.local)
set -e

REPO="${OMATTS_REPO:-parnoldx/omatts}"
VERSION="${OMATTS_VERSION:-latest}"
URL="${OMATTS_URL:-https://github.com/$REPO/releases/download/$VERSION/omatts-linux-x86_64.tar.zst}"
PREFIX="${PREFIX:-$HOME/.local}"
SHARE="$PREFIX/share/omatts"

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v zstd >/dev/null || { echo "zstd is required"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading omatts from $URL ..."
curl -fL --progress-bar -o "$TMP/omatts.tar.zst" "$URL"

echo "Extracting..."
tar --zstd -xf "$TMP/omatts.tar.zst" -C "$TMP"

mkdir -p "$PREFIX/bin" "$SHARE"
install -m755 "$TMP/omatts" "$PREFIX/bin/omatts"
rm -rf "$SHARE/models" "$SHARE/voices"
cp -r "$TMP/models" "$TMP/voices" "$SHARE/"

echo
echo "Installed omatts to $PREFIX/bin/omatts"
echo "Models and voices in $SHARE"
echo "Try it: omatts Hello world"
