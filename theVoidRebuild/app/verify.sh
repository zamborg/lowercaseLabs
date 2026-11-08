#!/bin/bash
# One command: run every flow we can without Zubin and write build/verify-report.md.
#   ./verify.sh          Mac harness (real speech + tagger + journal + dots on a fake cloud) and simulator UI flows
# Exits non-zero if anything failed. The report always ends with what this run did not cover.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p build; REPORT="$PWD/build/verify-report.md"; SHOTS="$PWD/build/shots"; rm -rf "$SHOTS"
SIM=$(xcrun simctl list devices booted | awk -F'[()]' '/iPhone/ {print $2; exit}')
[ -n "$SIM" ] || { xcrun simctl boot "iPhone 17 Pro" 2>/dev/null; SIM=$(xcrun simctl list devices booted | awk -F'[()]' '/iPhone/ {print $2; exit}'); }
FAIL=0

echo "== Mac harness"; ./validate.sh > build/harness.log 2>&1; HARNESS=$?; tail -25 build/harness.log
[ $HARNESS -eq 0 ] || FAIL=1

echo "== Simulator UI flows"
xcrun simctl uninstall "$SIM" com.lowercaselabs.void >/dev/null 2>&1   # fresh install: D5 default must hold
export VOID_ICLOUD=false VOID_ENTITLEMENTS= TEST_RUNNER_VOID_SHOTS="$SHOTS"
xcodegen generate >/dev/null
xcodebuild test -project TheVoid.xcodeproj -scheme TheVoid -destination "platform=iOS Simulator,id=$SIM" \
  -derivedDataPath build/sim -skipMacroValidation -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO > build/uitest.log 2>&1; UI=$?
grep -E "Test Case.*(passed|failed)|error:|TEST (SUCCEEDED|FAILED)" build/uitest.log | sed 's/^.*Test Case/Test Case/'
[ $UI -eq 0 ] || FAIL=1

{
  echo "# Verify report"; echo; echo "Run: $(date '+%Y-%m-%d %H:%M')  ·  $([ $FAIL -eq 0 ] && echo PASS || echo FAIL)"; echo
  echo "## Mac harness (real SpeechAnalyzer, real Tagger, real Journal; Release; fixtures instead of the mic)"; echo '```'; tail -25 build/harness.log; echo '```'
  echo "## Simulator UI flows (real UI and SwiftData; seeded entries, no speech/MLX)"; echo '```'
  grep -E "Test Case.*(passed|failed)|TEST (SUCCEEDED|FAILED)" build/uitest.log | sed 's/^.*Test Case/Test Case/'; echo '```'
  echo "Screenshots: build/shots/"; echo
  cat <<'NOT'
## What this run did not cover
- Recording from the real mic, the circle's feel, the lock gesture and touch latency: only the phone can show these. Last phone measurement is in the commit history, not this run.
- Capture UI flows in the simulator (no SpeechAnalyzer or MLX there); the pipeline is covered by the Mac harness, the screen by the phone.
- CloudKit: dots run against an in-memory stand-in. Real sharing between two accounts is unverified (needs SETUP steps 1 and 2).
- Tag accuracy on real messages (4 synthetic samples only, floor 2/4).
- Playback audio output and the audio session change (needs a device).
NOT
} > "$REPORT"
echo; echo "Report: $REPORT"; [ $FAIL -eq 0 ] && echo "VERIFY PASS" || echo "VERIFY FAIL"
exit $FAIL
