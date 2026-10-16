# GazeCam — menu-bar app + CoreMediaIO camera extension (virtual camera).
# System extensions only activate from /Applications, so `make dev` installs there.
APP        := GazeCam
CONFIG     ?= Debug
DERIVED    := build/DerivedData
APP_PATH   := $(DERIVED)/Build/Products/$(CONFIG)/$(APP).app
INSTALL_TO ?= /Applications
# No Xcode account on this Mac: profiles come via the App Store Connect API key.
ASC_KEY_ID    ?= 4YD9W3DQDH
ASC_ISSUER_ID ?= 44e7e5a5-69f6-465a-869d-901204a41013
ASC_AUTH := -authenticationKeyPath $(HOME)/.appstoreconnect/private_keys/AuthKey_$(ASC_KEY_ID).p8 -authenticationKeyID $(ASC_KEY_ID) -authenticationKeyIssuerID $(ASC_ISSUER_ID)

.PHONY: all generate build install run dev logs clean

all: build

generate:
	xcodegen generate

build: generate
	xcodebuild -project $(APP).xcodeproj -scheme $(APP) -configuration $(CONFIG) \
		-derivedDataPath $(DERIVED) -destination 'platform=macOS,arch=arm64' \
		-allowProvisioningUpdates $(ASC_AUTH) -quiet build
	@echo "Built: $(APP_PATH)"

install: build
	@pkill -x $(APP) 2>/dev/null || true
	rm -rf "$(INSTALL_TO)/$(APP).app"
	ditto "$(APP_PATH)" "$(INSTALL_TO)/$(APP).app"

run:
	open "$(INSTALL_TO)/$(APP).app"

dev: install run

logs:
	log stream --predicate 'subsystem BEGINSWITH "dev.iamhero.GazeCam" OR process == "GazeCam"' --level debug

clean:
	rm -rf build $(APP).xcodeproj
