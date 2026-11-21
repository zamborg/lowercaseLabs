# Router (Phase 0 spike)

Name: `Controllr · Router`

1. Dictate Text → `Dictated Text`
2. Controllr: Decide (Transcript = Dictated Text) → `Result`  (swap for "Decide (network spike)" to test Keychain/network)
3. Get Dictionary from Input (Input = Result) → `Decision`
4. Get Dictionary Value `action` from Decision
5. If `action` is `run`:
   - Get Dictionary Value `shortcut` from Decision → `Name`
   - Get Dictionary Value `input` from Decision → `Payload`
   - Run Shortcut (shortcut = **variable `Name`**, input = `Payload`)
   Otherwise: Get Dictionary Value `speak` → Speak Text

## Test module: `Controllr · Spike`
1. Get Dictionary from Input
2. Get Dictionary Value `echo`
3. Show Alert (title "Spike", message = echo)

Fallback if Run Shortcut by variable fails: If / Otherwise If chain on `Name` per module.
