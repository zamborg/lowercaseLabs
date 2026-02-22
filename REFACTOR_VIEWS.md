# Refactor Views Plan

## Why This Refactor
- `theVoidLocal/theVoid/theVoid/ContentView.swift` contains too many responsibilities.
- Large single-file UI increases merge conflicts and slows feature work.
- Splitting by feature will make ownership, navigation, and testing cleaner.

## Refactor Principles
1. Keep behavior and UX unchanged in early phases.
2. Refactor in small, incremental steps.
3. Build after every extraction step.
4. Move structure first, then improve internals.

## Target Structure
- `Features/Onboarding/OnboardingView.swift`
- `Features/Void/VoidExperienceView.swift`
- `Features/Journal/JournalView.swift`
- `Features/Social/SocialView.swift`
- `Features/Settings/SettingsView.swift`
- `Features/ModelSetup/LiquidModelPreparationView.swift`
- `Shared/Components/PulsingSignalDot.swift`
- `Shared/Components/TagChip.swift`
- `Shared/Components/EmotionAtlasView.swift`
- `Shared/Components/RecordingDecisionSheet.swift`
- `AppModel+Auth.swift`
- `AppModel+Entries.swift`
- `AppModel+Social.swift`
- `AppModel+ModelPreparation.swift`
- `Core/Networking/BackendClient.swift`
- `Core/Audio/RecorderEngine.swift`
- `Core/Notifications/NotificationScheduler.swift`
- `Core/Models/APIModels.swift`

## Phased Execution
1. Extract top-level feature views into separate files.
2. Extract shared UI components used by multiple screens.
3. Split `AppModel` into domain-based extensions.
4. Move services and API/data models out of UI files.
5. Remove dead code, stabilize imports, and run full build checks.

## Recommended Order
1. `LiquidModelPreparationView`
2. `OnboardingView`
3. `VoidExperienceView`
4. `JournalView` and atlas pieces
5. `SocialView`
6. `SettingsView`
7. Shared components
8. `AppModel` extensions
9. Service/model extraction

## Definition of Done
- `ContentView.swift` is a thin composition/root-routing file.
- Feature views and shared components are isolated by file.
- Build passes after each phase.
- No functional regressions in onboarding, recording, analysis, journal, social, and settings flows.

## Risks and Mitigations
- Risk: break `@EnvironmentObject`/binding flow during extraction.
  - Mitigation: move one screen at a time and compile immediately.
- Risk: hidden coupling in local private helpers.
  - Mitigation: promote only required helpers and keep scope tight.
- Risk: accidental behavior changes while moving code.
  - Mitigation: no logic edits during structural phases.
