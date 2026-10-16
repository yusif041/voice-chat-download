# Build Voice-chat-for-everyone

Install the runtime dependencies and PyInstaller in a Python environment:

```text
python -m pip install -r requirements-build.txt
```

## Windows

From PowerShell, run `powershell -ExecutionPolicy Bypass -File .\build-windows.ps1`. The execution-policy override applies only to this build process. The standalone executable is created at:

```text
dist\windows\Voice-chat-for-everyone.exe
```

## macOS

Build on a Mac with the matching Python architecture. PyInstaller does not cross-build macOS applications from Windows, and each Mac architecture needs its own native build.

```sh
python3 -m pip install -r requirements-build.txt
sh ./build-macos.sh arm64
```

For Intel Macs, use an x86_64 Python environment and run:

```sh
sh ./build-macos.sh x86_64
```

The `.app` bundles are created in `dist/macos-arm64` and `dist/macos-x86_64`. The first launch may require approving the app in macOS Privacy & Security because the build is not signed or notarized. Microphone access is needed for voice chat.
