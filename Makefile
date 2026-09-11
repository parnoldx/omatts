BUILD_DIR := .build
DIST_DIR  ?= dist
TARBALL   := $(DIST_DIR)/omatts-linux-x86_64.tar.zst

# Where make deploy publishes the tarball (GitHub Releases).
REPO ?= parnoldx/omatts
TAG  ?= v$(shell date +%Y.%m.%d)

.PHONY: build install shared deploy clean

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

# One distributable: binary + ONNX models (default variant from export_onnx.py)
# + voices. End users install it with the committed install.sh (curl | sh),
# which unpacks binary -> ~/.local/bin, models/voices -> ~/.local/share/omatts.
deploy: build
	.venv/bin/python export_onnx.py
	@mkdir -p $(DIST_DIR)
	tar --zstd --exclude='voices/.cache' --exclude='models/.cache' \
		-cf $(TARBALL) omatts models voices README.md
	ls -lh $(TARBALL)
	gh release create $(TAG) --repo $(REPO) --generate-notes $(TARBALL) \
		|| gh release upload $(TAG) --repo $(REPO) $(TARBALL) --clobber

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
