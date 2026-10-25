#!/bin/bash
# Builds Anybox.app and the disk image it ships in, target/release/anybox-VERSION.dmg.
#
# The app holds the window (anybox-gui), the command-line program (anybox) beside it, and the
# runtime both link titles with, in Contents/Resources/runtime. The window's "Install Command
# Line Tool" menu item links `anybox` on the PATH to the program inside the app.
#
# It is signed ad hoc unless SIGN_IDENTITY names a signing identity (a Developer ID, for a
# disk image other Macs will open).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/target/release"
VERSION=$(awk -F'"' '/^\[workspace.package\]/{p=1} p && /^version/{print $2; exit}' "$ROOT/Cargo.toml")
APP="$OUT/Anybox.app"
DMG="$OUT/anybox-$VERSION.dmg"
IDENTITY=${SIGN_IDENTITY:--}
# Notarization needs a secure timestamp; an ad hoc signature can't have one.
if [ "$IDENTITY" = "-" ]; then TIMESTAMP=--timestamp=none; else TIMESTAMP=--timestamp; fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" -p runtime -p anybox-cli -p anybox-gui

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/runtime"
cp "$OUT/anybox-gui" "$OUT/anybox" "$APP/Contents/MacOS/"
cp "$OUT/libruntime.a" "$OUT/signatures.txt" "$APP/Contents/Resources/runtime/"

# The icon, at every size macOS asks for.
ICONSET=$(mktemp -d)/AppIcon.iconset
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z $size $size "$ROOT/crates/anybox-gui/assets/icon.png" \
    --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z $double $double "$ROOT/crates/anybox-gui/assets/icon.png" \
    --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"

cat >"$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>anybox</string>
  <key>CFBundleExecutable</key><string>anybox-gui</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>io.github.aovestdipaperino.anybox</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>anybox</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

# The program beside the window is signed first, then the app around it.
codesign --force --options runtime "$TIMESTAMP" -s "$IDENTITY" "$APP/Contents/MacOS/anybox"
codesign --force --options runtime "$TIMESTAMP" -s "$IDENTITY" "$APP"
codesign --verify --strict "$APP"

STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -quiet -volname "anybox $VERSION" -srcfolder "$STAGE" -format UDZO "$DMG"
rm -rf "$STAGE"
echo "wrote $DMG"
