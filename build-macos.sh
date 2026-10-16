#!/bin/sh
set -eu

if [ "$#" -ne 1 ] || { [ "$1" != "arm64" ] && [ "$1" != "x86_64" ]; }; then
    echo "Usage: ./build-macos.sh arm64|x86_64" >&2
    exit 2
fi

target_arch="$1"
python_arch="$(python3 -c 'import platform; print(platform.machine())')"
if [ "$python_arch" != "$target_arch" ]; then
    echo "A $target_arch Python is required; current Python is $python_arch." >&2
    exit 1
fi

root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python3 -m PyInstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "Voice-chat-for-everyone" \
    --osx-bundle-identifier "com.voicechatforeveryone.app" \
    --target-architecture "$target_arch" \
    --collect-data ursina \
    --collect-binaries panda3d \
    --collect-data panda3d \
    --collect-all _sounddevice_data \
    --specpath "$root/build/macos-$target_arch" \
    --distpath "$root/dist/macos-$target_arch" \
    --workpath "$root/build/macos-$target_arch" \
    "$root/voice-chat-3D.py"

app_bundle="$root/dist/macos-$target_arch/Voice-chat-for-everyone.app"
info_plist="$app_bundle/Contents/Info.plist"
if /usr/libexec/PlistBuddy -c "Print :NSMicrophoneUsageDescription" "$info_plist" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Set :NSMicrophoneUsageDescription Voice chat needs microphone access." "$info_plist"
else
    /usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string Voice chat needs microphone access." "$info_plist"
fi
