#!/bin/sh
# omatts installer
#   binary       -> ~/.local/bin/omatts
#   models       -> ~/.local/share/omatts/models       (EN pack, default)
#   voices       -> ~/.local/share/omatts/voices       (EN voices + <lang>/ from packs)
#   models-<tag> -> ~/.local/share/omatts/models-<tag> (language packs)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/parnoldx/omatts/master/install.sh | sh
#
# Optional overrides (env vars):
#   OMATTS_REPO=parnoldx/omatts   GitHub repo (must match this script's home)
#   OMATTS_VERSION=latest         release tag, or e.g. v2026.09.11
#   OMATTS_URL=...                direct tarball URL, skips repo/version
#   OMATTS_PACKS="de"             language packs to install in addition
#   OMATTS_PACKS_ONLY=1           install only the packs (into an existing install)
#   PREFIX=/usr/local             install root (default ~/.local)
#
# Examples:
#   fresh install with German:      curl -fsSL .../install.sh | sh   # answer y at the prompt
#   non-interactive, with German:   curl -fsSL .../install.sh | OMATTS_PACKS=de sh
#   add German to an existing one:  curl -fsSL .../install.sh | OMATTS_PACKS=de OMATTS_PACKS_ONLY=1 sh
set -e

REPO="${OMATTS_REPO:-parnoldx/omatts}"
VERSION="${OMATTS_VERSION:-latest}"
# "latest" uses GitHub's moving asset URL (no such tag exists); a pinned
# version uses its release tag. Pack assets live in the same place.
if [ "$VERSION" = latest ]; then
	BASE="https://github.com/$REPO/releases/latest/download"
else
	BASE="https://github.com/$REPO/releases/download/$VERSION"
fi
URL="${OMATTS_URL:-$BASE/omatts-linux-x86_64.tar.zst}"
# OMATTS_URL also moves the pack base with it
PACK_BASE="$BASE"
if [ -n "$OMATTS_URL" ]; then PACK_BASE="${OMATTS_URL%/*}"; fi
PREFIX="${PREFIX:-$HOME/.local}"
SHARE="$PREFIX/share/omatts"
PACKS="${OMATTS_PACKS:-}"

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v zstd >/dev/null || { echo "zstd is required"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch() { curl -fL --progress-bar -o "$2" "$1"; }

if [ -n "$OMATTS_PACKS_ONLY" ]; then
	:
else
	echo "Downloading omatts from $URL ..."
	fetch "$URL" "$TMP/omatts.tar.zst"

	echo "Extracting..."
	tar --zstd -xf "$TMP/omatts.tar.zst" -C "$TMP"

	mkdir -p "$PREFIX/bin" "$SHARE"
	install -m755 "$TMP/omatts" "$PREFIX/bin/omatts"
	# keep German voices across reinstalls: they come from the de pack, which
	# the main tarball doesn't contain and this run may not download
	if [ -d "$SHARE/voices/de" ]; then
		mv "$SHARE/voices/de" "$TMP/keep-de"
		KEEP_DE=1
	fi
	rm -rf "$SHARE/models" "$SHARE/voices"
	cp -r "$TMP/models" "$TMP/voices" "$SHARE/"
	if [ -n "$KEEP_DE" ]; then
		mkdir -p "$SHARE/voices"
		mv "$TMP/keep-de" "$SHARE/voices/de"
	fi

	echo
	echo "Installed omatts to $PREFIX/bin/omatts"
	echo "Models and voices in $SHARE"
	echo "Try it: omatts Hello world"

	# Ask about language packs unless the caller already picked some.
	# Read from /dev/tty: under `curl | sh` stdin is the script itself.
	# No tty (CI, piped) -> default to no, never hang.
	if [ -z "$PACKS" ] && (exec 3</dev/tty) 2>/dev/null; then
		if [ -e "$SHARE/models-de" ]; then
			prompt="German language pack already installed — update it? [y/N] "
		else
			prompt="Also install the German language pack (~120 MB)? [y/N] "
		fi
		printf "%s" "$prompt"
		read answer < /dev/tty 2>/dev/null || answer=
		case $answer in [yY]|[yY][eE][sS]) PACKS=de ;; esac
	fi
fi

for pack in $PACKS; do
	echo "Downloading language pack '$pack' ..."
	fetch "$PACK_BASE/omatts-$pack-pack.tar.zst" "$TMP/pack.tar.zst"

	rm -rf "$TMP/pack"; mkdir "$TMP/pack"
	tar --zstd -xf "$TMP/pack.tar.zst" -C "$TMP/pack"

	mkdir -p "$SHARE/voices"
	for dir in "$TMP/pack"/models-*; do
		[ -d "$dir" ] || continue
		cp -r "$dir" "$SHARE/"
	done
	for dir in "$TMP/pack"/voices/*/; do
		[ -d "$dir" ] || continue
		cp -r "$dir" "$SHARE/voices/"
	done
	echo "Installed language pack '$pack' in $SHARE"
done
