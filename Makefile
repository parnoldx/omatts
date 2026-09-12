BUILD_DIR := .build
DIST_DIR  ?= dist
TARBALL   := $(DIST_DIR)/omatts-linux-x86_64.tar.zst
DE_PACK   := $(DIST_DIR)/omatts-de-pack.tar.zst

# Where make deploy publishes the tarball (GitHub Releases).
REPO ?= parnoldx/omatts
TAG  ?= v$(shell date +%Y.%m.%d)

.PHONY: build install shared deploy clean test test-e2e site-check site-serve

PREFIX ?= $(HOME)/.local

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$$(nproc)

# shared lib for FFI, output next to the CLI
shared:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIB=ON
	cmake --build $(BUILD_DIR) -j$$(nproc)

install: build
	install -Dm755 omatts $(DESTDIR)$(PREFIX)/bin/omatts

# Unit tests (pure logic, no models needed)
test: build
	cmake --build $(BUILD_DIR) -j$$(nproc) --target omatts-test
	ctest --test-dir $(BUILD_DIR) -R omatts-test --output-on-failure

# End-to-end tests: drive the built binary with real models and a real voice
# through the CLI and HTTP server. Auto-skips if models/ is absent.
test-e2e: build
	ctest --test-dir $(BUILD_DIR) -R omatts-e2e --output-on-failure

# One distributable: binary + ONNX models (default variant from export_onnx.py)
# + voices. End users install it with the committed install.sh (curl | sh),
# which unpacks binary -> ~/.local/bin, models/voices -> ~/.local/share/omatts.
# Two release assets:
#   main tarball  = binary + EN pack (models/ + English voices)
#   de pack       = models-de/ + voices/de/, installed by install.sh via
#                   OMATTS_PACKS=de (pack tarballs are named omatts-<tag>-pack.tar.zst)
deploy: build
	.venv/bin/python export_onnx.py
	@mkdir -p $(DIST_DIR)
	tar --zstd --exclude='voices/.cache' --exclude='models/.cache' --exclude='voices/de' \
		-cf $(TARBALL) omatts models voices README.md
	tar --zstd --exclude='models-de/.cache' --exclude='voices/.cache' -cf $(DE_PACK) models-de voices/de
	ls -lh $(TARBALL) $(DE_PACK)
	gh release create $(TAG) --repo $(REPO) --generate-notes $(TARBALL) $(DE_PACK) \
		|| gh release upload $(TAG) --repo $(REPO) $(TARBALL) $(DE_PACK) --clobber

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)

# Validate the self-contained static documentation website.
site-check:
	python3 scripts/check-site.py

# Serve the static website locally for testing.
site-serve:
	python3 -m http.server -d site 8080
