# Veil 0.1.0 development build

[Download Veil for macOS](Veil-0.1.0-macos-universal.zip)

The archive contains `Veil.app` with both Apple Silicon (`arm64`) and Intel (`x86_64`) executables, targeting macOS 13 or later. Both architectures compile; live TextEdit validation was performed on Apple Silicon running macOS 26.5.2. Intel and older macOS versions have not been runtime-tested.

This is a development build, ad-hoc signed and not Apple-notarized. It is intended for personal testing; it is not a frictionless public installer. If macOS does not allow the downloaded build to launch, build locally from the [source instructions](../README.md) instead. Do not disable system security protections.

On each Mac:

1. Extract the archive and put `Veil.app` in a stable location, such as Applications.
2. Open it and grant Veil access in System Settings → Privacy & Security → Accessibility.
3. Select **Load TextEdit demo**, then **Start masking**. Focus TextEdit manually with the synthetic fixture from the main README.
4. After confirming the demo works, configure a target bundle ID and your own labeled secrets. Configuration is session-only and must be entered on each launch.

The source still contains the original text. This app adds reactive desktop masks and does not guarantee that a stream cannot capture a frame before masking. Window-only capture can omit the masks. iTerm2 compatibility is unverified.

Verify the archive with `shasum -a 256 -c SHA256SUMS` from this directory. To reproduce a universal archive, run `bash scripts/package-universal.sh` from `veil/` on a Mac with Xcode installed. This command packages a separate app and does not overwrite the already-tested local `build/Veil.app`.
