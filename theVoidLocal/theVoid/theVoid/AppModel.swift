import AVFoundation
import Foundation
import SwiftUI

// MARK: - App enums

enum SubmissionState: String {
    case idle
    case recording
    case uploading
    case transcribing
    case insightsReady
    case failed

    var title: String {
        switch self {
        case .idle:
            return "Ready"
        case .recording:
            return "Recording"
        case .uploading:
            return "Uploading"
        case .transcribing:
            return "Transcribing"
        case .insightsReady:
            return "Insights Ready"
        case .failed:
            return "Failed"
        }
    }
}

private enum AppModelError: LocalizedError {
    case missingLocalUserIdentity

    var errorDescription: String? {
        switch self {
        case .missingLocalUserIdentity:
            return "Missing local user identity."
        }
    }
}

// MARK: - App model

@MainActor
final class AppModel: ObservableObject {
    private static let socialHistoryLimit = 300

    @Published var appleUserID: String = ""
    @Published var userID: String = ""
    @Published var displayName: String = ""
    @Published var anonymousHandle: String = ""
    @Published var dailyCheckin: Date = Date()
    @Published var timezone: String = TimeZone.current.identifier
    @Published var notificationEnabled: Bool = false
    @Published var reminderStatus: String?
    @Published var reminderWeekdays: Set<Int> = Set(ReminderWeekday.ordered.map(\.rawValue))
    @Published var reminderTimesByWeekday: [Int: Date] = [:]
    @Published var onboardingComplete: Bool = false

    @Published var entries: [APIEntry] = []
    @Published var socialDots: [APISocialDot] = []
    @Published var pendingDrafts: [URL] = []
    @Published var submissionState: SubmissionState = .idle
    @Published var activeEntryID: String?
    @Published var lastInsightRuntimeSummary: String?
    @Published var liquidModelPrepared: Bool = false
    @Published var showsLiquidModelPreparationScreen: Bool = false
    @Published var isPreparingLiquidModel: Bool = false
    @Published var liquidModelPreparationProgress: Double = 0
    @Published var liquidModelPreparationStatus: String = "Preparing on-device AI model..."
    @Published var liquidModelPreparationError: String?

    @Published var transcriptionEngine: TranscriptionEngineKind = .defaultChoice
    @Published var transcriptionLanguage: TranscriptionLanguage = .device
    @Published var isTranscriptionEngineReady: Bool = false
    @Published var isPreparingTranscriptionEngine: Bool = false
    @Published var transcriptionEnginePreparationError: String?
    @Published var liveTranscript: String = ""
    @Published var isTranscribingForReview: Bool = false
    @Published var healthAuthorizationState: HealthAuthorizationState = .notDetermined
    @Published var liveHealthSnapshot: EntryHealthSnapshot?
    @Published var isFetchingLiveHealthSnapshot: Bool = false
    @Published var healthIntegrationEnabled: Bool = true
    @Published var iCloudSyncEnabled: Bool = true
    @Published var iCloudSyncStatus: ICloudSyncStatus = .idle
    @Published var iCloudLastSyncAt: Date?

    @Published var inviteURL: String?
    @Published var inviteToken: String?
    @Published var errorMessage: String?

    private let socialClient: any SocialFeatureClient
    private let draftStore = DraftStore()
    private let localStore = LocalJournalStore()
    private let healthKitManager = HealthKitManager()
    private let iCloudSyncService: ICloudSyncService
    private var liquidModelPreparationTask: Task<Void, Never>?
    private var liquidModelPreparationOperationID: UUID?
    private let transcriptionEngineCoordinator = TranscriptionEngineCoordinator()
    private var transcriptionEnginePreparationTask: Task<Void, Never>?
    private var transcriptionEnginePreparationOperationID: UUID?
    private var transcriptionForReviewTask: Task<Void, Never>?

    private enum Keys {
        static let appleUserID = "thevoid.appleUserID"
        static let userID = "thevoid.userID"
        static let displayName = "thevoid.displayName"
        static let anonymousHandle = "thevoid.anonymousHandle"
        static let dailyCheckin = "thevoid.dailyCheckin"
        static let timezone = "thevoid.timezone"
        static let notificationEnabled = "thevoid.notificationEnabled"
        static let reminderWeekdays = "thevoid.reminderWeekdays"
        static let reminderTimesByWeekday = "thevoid.reminderTimesByWeekday"
        static let onboardingComplete = "thevoid.onboardingComplete"
        static let liquidModelPrepared = "thevoid.liquidModelPrepared"
        static let transcriptionEngine = "thevoid.transcription.engine"
        static let transcriptionLanguage = "thevoid.transcription.language"
        static let healthIntegrationEnabled = "thevoid.healthkit.enabled"
        static let iCloudSyncEnabled = "thevoid.icloud.sync.enabled"
        static let iCloudBootstrapSyncedUsers = "thevoid.icloud.bootstrap.synced_users"
    }

    var needsOnboarding: Bool {
        userID.isEmpty || !onboardingComplete
    }

    var iCloudSyncStatusText: String {
        switch iCloudSyncStatus {
        case .disabled:
            return "Disabled"
        case .unavailable:
            return "Unavailable"
        case .idle:
            return "Idle"
        case .syncing:
            return "Syncing..."
        case .error(let message):
            return "Error: \(message)"
        }
    }

    private var currentSocialProfile: SocialProfile? {
        guard !userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return SocialProfile(
            userID: userID,
            displayName: displayName.isEmpty ? nil : displayName,
            anonymousHandle: anonymousHandle.isEmpty ? "void-\(userID.prefix(8))" : anonymousHandle
        )
    }

    init(socialClient: any SocialFeatureClient = CloudKitSocialFeatureClient()) {
        self.socialClient = socialClient
        iCloudSyncService = ICloudSyncService(localJournalStore: localStore, draftStore: draftStore)
        loadPersistedState()
        Task {
            await iCloudSyncService.setStatusHandler { [weak self] status, lastSync in
                guard let self else { return }
                self.iCloudSyncStatus = status
                self.iCloudLastSyncAt = lastSync
            }
        }
        reloadDrafts()
        if !userID.isEmpty {
            Task {
                await configureICloudSyncForCurrentUser()
                await refreshAll()
            }
        }
        if healthIntegrationEnabled {
            Task {
                await refreshHealthAuthorizationState()
                await refreshLiveHealthSnapshot()
            }
        }
        prepareTranscriptionEngineIfNeeded()
    }

    func loadPersistedState() {
        let defaults = UserDefaults.standard
        appleUserID = defaults.string(forKey: Keys.appleUserID) ?? ""
        userID = defaults.string(forKey: Keys.userID) ?? ""
        displayName = defaults.string(forKey: Keys.displayName) ?? ""
        anonymousHandle = defaults.string(forKey: Keys.anonymousHandle) ?? ""
        timezone = defaults.string(forKey: Keys.timezone) ?? TimeZone.current.identifier
        notificationEnabled = defaults.object(forKey: Keys.notificationEnabled) as? Bool ?? false
        onboardingComplete = defaults.bool(forKey: Keys.onboardingComplete)

        if let timeString = defaults.string(forKey: Keys.dailyCheckin),
           let parsed = DateFormatter.hhmm.date(from: timeString) {
            dailyCheckin = parsed
        }

        if let storedWeekdays = defaults.array(forKey: Keys.reminderWeekdays) as? [Int], !storedWeekdays.isEmpty {
            reminderWeekdays = Set(storedWeekdays.filter { ReminderWeekday.ordered.map(\.rawValue).contains($0) })
        } else {
            reminderWeekdays = Set(ReminderWeekday.ordered.map(\.rawValue))
        }

        reminderTimesByWeekday = Self.defaultReminderTimes(base: dailyCheckin)
        if let storedTimes = defaults.dictionary(forKey: Keys.reminderTimesByWeekday) as? [String: String] {
            for (weekdayKey, timeString) in storedTimes {
                guard let weekday = Int(weekdayKey),
                      ReminderWeekday.ordered.map(\.rawValue).contains(weekday),
                      let parsed = DateFormatter.hhmm.date(from: timeString) else {
                    continue
                }
                reminderTimesByWeekday[weekday] = parsed
            }
        }

        liquidModelPrepared = defaults.bool(forKey: Keys.liquidModelPrepared)
        if let rawEngine = defaults.string(forKey: Keys.transcriptionEngine),
           let storedEngine = TranscriptionEngineKind(rawValue: rawEngine),
           storedEngine.isAvailable {
            transcriptionEngine = storedEngine
        } else {
            transcriptionEngine = .defaultChoice
        }
        if let rawLanguage = defaults.string(forKey: Keys.transcriptionLanguage),
           let storedLanguage = TranscriptionLanguage(rawValue: rawLanguage) {
            transcriptionLanguage = storedLanguage
        } else {
            transcriptionLanguage = .device
        }
        if defaults.object(forKey: Keys.healthIntegrationEnabled) != nil {
            healthIntegrationEnabled = defaults.bool(forKey: Keys.healthIntegrationEnabled)
        } else {
            healthIntegrationEnabled = true
        }
        if defaults.object(forKey: Keys.iCloudSyncEnabled) != nil {
            iCloudSyncEnabled = defaults.bool(forKey: Keys.iCloudSyncEnabled)
        } else {
            iCloudSyncEnabled = true
        }
    }

    func persistState() {
        let defaults = UserDefaults.standard
        defaults.set(appleUserID, forKey: Keys.appleUserID)
        defaults.set(userID, forKey: Keys.userID)
        defaults.set(displayName, forKey: Keys.displayName)
        defaults.set(anonymousHandle, forKey: Keys.anonymousHandle)
        defaults.set(DateFormatter.hhmm.string(from: dailyCheckin), forKey: Keys.dailyCheckin)
        defaults.set(timezone, forKey: Keys.timezone)
        defaults.set(notificationEnabled, forKey: Keys.notificationEnabled)
        defaults.set(Array(reminderWeekdays).sorted(), forKey: Keys.reminderWeekdays)
        let reminderTimesPayload = reminderTimesByWeekday.reduce(into: [String: String]()) { partialResult, item in
            partialResult[String(item.key)] = DateFormatter.hhmm.string(from: item.value)
        }
        defaults.set(reminderTimesPayload, forKey: Keys.reminderTimesByWeekday)
        defaults.set(onboardingComplete, forKey: Keys.onboardingComplete)
        defaults.set(healthIntegrationEnabled, forKey: Keys.healthIntegrationEnabled)
        defaults.set(iCloudSyncEnabled, forKey: Keys.iCloudSyncEnabled)
        defaults.set(transcriptionEngine.rawValue, forKey: Keys.transcriptionEngine)
        defaults.set(transcriptionLanguage.rawValue, forKey: Keys.transcriptionLanguage)
    }

    func signIn(appleUserID: String, suggestedName: String?) async {
        do {
            try await AppleICloudIdentity.requireAvailableICloudAccount()

            self.appleUserID = appleUserID
            if userID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                userID = AppleICloudIdentity.localUserID(for: appleUserID)
            }
            if displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               let suggestedName,
               !suggestedName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                displayName = suggestedName
            }
            if anonymousHandle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                anonymousHandle = AppleICloudIdentity.anonymousHandle(for: appleUserID)
            }

            persistState()
            await configureICloudSyncForCurrentUser()
            await refreshAll()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

#if DEBUG
    func signInDev(localIdentifier: String?) async {
        let trimmed = localIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let baseToken = trimmed.isEmpty ? UUID().uuidString : trimmed
        let localAppleUserID = baseToken.hasPrefix("dev-") ? baseToken : "dev-\(baseToken)"
        await signIn(appleUserID: localAppleUserID, suggestedName: nil)
    }
#endif

    func completeOnboarding() {
        onboardingComplete = true
        persistState()
    }

    func saveProfile() async {
        if anonymousHandle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            anonymousHandle = appleUserID.isEmpty
                ? "void-\(UUID().uuidString.prefix(8).lowercased())"
                : AppleICloudIdentity.anonymousHandle(for: appleUserID)
        }
        persistState()
    }

    func signOut() {
        transcriptionEnginePreparationTask?.cancel()
        transcriptionEnginePreparationTask = nil
        transcriptionEnginePreparationOperationID = nil
        isPreparingTranscriptionEngine = false
        isTranscriptionEngineReady = false
        Task {
            await transcriptionEngineCoordinator.unload()
        }
        liquidModelPreparationTask?.cancel()
        liquidModelPreparationTask = nil
        liquidModelPreparationOperationID = nil
        showsLiquidModelPreparationScreen = false
        isPreparingLiquidModel = false
        liquidModelPreparationProgress = 0
        liquidModelPreparationStatus = "Preparing on-device AI model..."
        liquidModelPreparationError = nil
        appleUserID = ""
        userID = ""
        displayName = ""
        anonymousHandle = ""
        reminderStatus = nil
        entries = []
        socialDots = []
        inviteURL = nil
        inviteToken = nil
        liveHealthSnapshot = nil
        healthAuthorizationState = .notDetermined
        isFetchingLiveHealthSnapshot = false
        iCloudLastSyncAt = nil
        iCloudSyncStatus = .idle
        onboardingComplete = false
        persistState()
    }

    func refreshAll() async {
        await configureICloudSyncForCurrentUser()
        await refreshEntries()
        await refreshSocialDots()
        reloadDrafts()
        await refreshHealthAuthorizationState()
        await refreshLiveHealthSnapshot()
    }

    func setICloudSyncEnabled(_ enabled: Bool) {
        iCloudSyncEnabled = enabled
        persistState()
        guard !userID.isEmpty else { return }
        Task {
            await iCloudSyncService.setEnabled(enabled, userID: userID)
            if enabled {
                await iCloudSyncService.syncNow(userID: userID, reason: "toggle_enable")
                await refreshEntries()
                reloadDrafts()
            }
        }
    }

    func syncNow() async {
        guard !userID.isEmpty else { return }
        await iCloudSyncService.syncNow(userID: userID, reason: "manual")
        await refreshEntries()
        reloadDrafts()
    }

    func handleAppDidBecomeActive() async {
        guard iCloudSyncEnabled, !userID.isEmpty else { return }
        await iCloudSyncService.syncNow(userID: userID, reason: "foreground")
        await refreshEntries()
        reloadDrafts()
    }

    func refreshEntries() async {
        guard !userID.isEmpty else {
            entries = []
            return
        }
        entries = localStore.listEntries(for: userID)
    }

    func refreshSocialDots() async {
        guard let profile = currentSocialProfile else {
            socialDots = []
            return
        }
        do {
            let envelope = try await socialClient.fetchSocialDots(
                for: profile,
                history: true,
                limit: Self.socialHistoryLimit
            )
            socialDots = envelope.dots.sorted(by: Self.socialDotIsMoreRecent)
        } catch is CancellationError {
            // Pull-to-refresh and view transitions can cancel in-flight tasks; keep last good state.
            return
        } catch {
            errorMessage = describeError(error)
        }
    }

    func reloadDrafts() {
        pendingDrafts = draftStore.listDrafts()
        guard !userID.isEmpty else { return }
        let draftIDs = pendingDrafts.map { $0.deletingPathExtension().lastPathComponent }.filter { !$0.isEmpty }
        Task {
            for draftID in draftIDs {
                await iCloudSyncService.queueDraftUpsert(userID: userID, draftID: draftID)
            }
        }
    }

    func makeDraftURL() -> URL {
        draftStore.makeDraftURL()
    }

    func deleteDraft(_ url: URL) {
        let draftID = url.deletingPathExtension().lastPathComponent
        draftStore.delete(url)
        if !userID.isEmpty, !draftID.isEmpty {
            Task {
                await iCloudSyncService.queueDraftDelete(userID: userID, draftID: draftID)
                await iCloudSyncService.syncNow(userID: userID, reason: "delete_draft")
            }
        }
        reloadDrafts()
    }

    func refreshHealthAuthorizationState() async {
        guard healthIntegrationEnabled else {
            healthAuthorizationState = .notDetermined
            liveHealthSnapshot = nil
            return
        }
        healthAuthorizationState = await healthKitManager.authorizationStatus()
    }

    func requestHealthAuthorization() async {
        guard healthIntegrationEnabled else {
            return
        }
        do {
            healthAuthorizationState = try await healthKitManager.requestReadAuthorization()
            if healthAuthorizationState.isAuthorized {
                await refreshLiveHealthSnapshot()
            } else if healthAuthorizationState == .denied {
                liveHealthSnapshot = nil
                errorMessage = "Health access is off. Open the Health app -> Sharing -> Apps -> theVoid to enable access."
            }
        } catch {
            errorMessage = "Could not request Health access: \(error.localizedDescription)"
            healthAuthorizationState = await healthKitManager.authorizationStatus()
        }
    }

    func refreshLiveHealthSnapshot() async {
        guard healthIntegrationEnabled else {
            liveHealthSnapshot = nil
            return
        }

        isFetchingLiveHealthSnapshot = true
        defer { isFetchingLiveHealthSnapshot = false }

        let snapshot = await captureHealthSnapshotForNote(at: Date())
        liveHealthSnapshot = snapshot
    }

    func captureHealthSnapshotForNote(at timestamp: Date) async -> EntryHealthSnapshot? {
        guard healthIntegrationEnabled else {
            return nil
        }
        if !healthAuthorizationState.isAuthorized {
            await refreshHealthAuthorizationState()
            guard healthAuthorizationState.isAuthorized else {
                return nil
            }
        }
        do {
            return try await fetchHealthSnapshotWithTimeout(
                at: timestamp,
                timeoutSeconds: 1.5
            )
        } catch {
            return nil
        }
    }

    func submitDraft(url: URL, durationSeconds: Int, shareToSocial: Bool = true, overrideTranscript: String? = nil) async {
        guard !userID.isEmpty else {
            errorMessage = "Sign in required"
            return
        }
        do {
            let normalizedDuration = max(1, min(durationSeconds, 300))
            let noteTimestamp = Date()
            let localDate = DateFormatter.localDate.string(from: noteTimestamp)
            submissionState = .transcribing
            lastInsightRuntimeSummary = nil

            async let analysisTask = LocalReflectionAnalyzer.analyze(
                audioURL: url,
                durationSeconds: normalizedDuration,
                useLiquidInsights: liquidModelPrepared,
                overrideTranscriptText: overrideTranscript,
                transcriptionConfiguration: TranscriptionConfiguration(
                    engine: transcriptionEngine,
                    language: transcriptionLanguage
                )
            )
            async let healthSnapshotTask = captureHealthSnapshotForNote(at: noteTimestamp)

            let ((transcript, insight, generatedTitle), healthSnapshot) = await (analysisTask, healthSnapshotTask)
            lastInsightRuntimeSummary = Self.insightRuntimeSummary(from: transcript?.providerMetadata)
            let stored = try localStore.saveReflection(
                for: userID,
                draftURL: url,
                durationSeconds: normalizedDuration,
                transcript: transcript,
                insight: insight,
                localDate: localDate,
                wasSharedToSocial: shareToSocial,
                healthSnapshot: healthSnapshot,
                title: generatedTitle
            )
            await iCloudSyncService.queueEntryUpsert(userID: userID, entryID: stored.id)
            if let healthSnapshot {
                liveHealthSnapshot = healthSnapshot
            }
            activeEntryID = stored.id
            submissionState = insight == nil ? .idle : .insightsReady

            if shareToSocial, let insight {
                do {
                    try await publishSocialDot(
                        localDate: localDate,
                        insight: insight
                    )
                } catch {
                    errorMessage = "Saved locally, but iCloud social sync failed: \(error.localizedDescription)"
                }
            } else if shareToSocial {
                errorMessage = "Saved audio note locally without tags. Retranscribe from the entry to generate dots."
            }

            await refreshEntries()
            await refreshSocialDots()
            reloadDrafts()
            await iCloudSyncService.syncNow(userID: userID, reason: "submit_draft")
        } catch {
            submissionState = .failed
            errorMessage = describeError(error)
            reloadDrafts()
        }
    }

    func redownloadLiquidModel() {
        startLiquidModelPreparation(forceRedownload: true)
    }

    func prepareLiquidModelIfNeeded() {
        startLiquidModelPreparation(forceRedownload: false)
    }

    func retryLiquidModelPreparation() {
        prepareLiquidModelIfNeeded()
    }

    func cancelLiquidModelPreparation() {
        liquidModelPreparationTask?.cancel()
        liquidModelPreparationTask = nil
        liquidModelPreparationOperationID = nil
        Task {
            await LocalReflectionAnalyzer.cancelLiquidModelPreparation()
        }
        isPreparingLiquidModel = false
        liquidModelPreparationError = nil
        liquidModelPreparationStatus = "Model download canceled."
        setLiquidModelPrepared(false)
        showsLiquidModelPreparationScreen = false
    }

    private func startLiquidModelPreparation(forceRedownload: Bool) {
        guard !userID.isEmpty, onboardingComplete else { return }
        if isPreparingLiquidModel { return }
        if !forceRedownload, liquidModelPrepared {
            showsLiquidModelPreparationScreen = false
            return
        }

        liquidModelPreparationTask?.cancel()
        liquidModelPreparationTask = nil
        liquidModelPreparationOperationID = nil

        if forceRedownload {
            setLiquidModelPrepared(false)
        }

        liquidModelPreparationError = nil
        liquidModelPreparationProgress = 0
        liquidModelPreparationStatus = forceRedownload
            ? "Redownloading on-device AI model..."
            : "Downloading on-device AI model..."
        showsLiquidModelPreparationScreen = true
        isPreparingLiquidModel = true
        let operationID = UUID()
        liquidModelPreparationOperationID = operationID

        liquidModelPreparationTask = Task { [weak self] in
            guard let self else { return }
            do {
                let descriptor = try await LocalReflectionAnalyzer.prepareLiquidModel(
                    forceRedownload: forceRedownload,
                    progressHandler: { progress in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.liquidModelPreparationProgress = max(0, min(1, progress))
                        }
                    }
                )
                await MainActor.run {
                    guard self.liquidModelPreparationOperationID == operationID else { return }
                    self.isPreparingLiquidModel = false
                    self.liquidModelPreparationProgress = 1
                    self.liquidModelPreparationStatus = "Model ready (\(descriptor))"
                    self.liquidModelPreparationError = nil
                    self.setLiquidModelPrepared(true)
                    self.showsLiquidModelPreparationScreen = false
                    self.liquidModelPreparationOperationID = nil
                }
            } catch {
                if error is CancellationError {
                    await MainActor.run {
                        guard self.liquidModelPreparationOperationID == operationID else { return }
                        self.isPreparingLiquidModel = false
                        self.liquidModelPreparationError = nil
                        self.liquidModelPreparationStatus = "Model download canceled."
                        self.setLiquidModelPrepared(false)
                        self.showsLiquidModelPreparationScreen = false
                        self.liquidModelPreparationOperationID = nil
                    }
                    return
                }

                await MainActor.run {
                    guard self.liquidModelPreparationOperationID == operationID else { return }
                    self.isPreparingLiquidModel = false
                    self.liquidModelPreparationStatus = "Model download failed."
                    self.liquidModelPreparationError = error.localizedDescription
                    self.setLiquidModelPrepared(false)
                    self.showsLiquidModelPreparationScreen = true
                    self.liquidModelPreparationOperationID = nil
                }
            }
        }
    }

    private func setLiquidModelPrepared(_ prepared: Bool) {
        liquidModelPrepared = prepared
        UserDefaults.standard.set(prepared, forKey: Keys.liquidModelPrepared)
    }

    // MARK: - Transcription engine

    var transcriptionEngineStatusText: String {
        if isTranscriptionEngineReady {
            return "Ready"
        }
        if isPreparingTranscriptionEngine {
            return "Preparing..."
        }
        return transcriptionEnginePreparationError == nil ? "Not ready" : "Needs attention"
    }

    func setTranscriptionEngine(_ engine: TranscriptionEngineKind) {
        guard engine.isAvailable, transcriptionEngine != engine else { return }
        transcriptionEngine = engine
        persistState()
        prepareTranscriptionEngineIfNeeded(force: true)
    }

    func setTranscriptionLanguage(_ language: TranscriptionLanguage) {
        guard transcriptionLanguage != language else { return }
        transcriptionLanguage = language
        persistState()
        prepareTranscriptionEngineIfNeeded(force: true)
    }

    func prepareTranscriptionEngineIfNeeded(force: Bool = false) {
        if !force, isTranscriptionEngineReady || isPreparingTranscriptionEngine {
            return
        }

        let previousPreparationTask = transcriptionEnginePreparationTask
        previousPreparationTask?.cancel()
        let operationID = UUID()
        transcriptionEnginePreparationOperationID = operationID
        isTranscriptionEngineReady = false
        isPreparingTranscriptionEngine = true
        transcriptionEnginePreparationError = nil
        let configuration = TranscriptionConfiguration(
            engine: transcriptionEngine,
            language: transcriptionLanguage
        )

        transcriptionEnginePreparationTask = Task { [weak self] in
            guard let self else { return }
            await previousPreparationTask?.value
            guard !Task.isCancelled else { return }
            do {
                try await transcriptionEngineCoordinator.prepare(configuration: configuration)
                guard !Task.isCancelled,
                      transcriptionEnginePreparationOperationID == operationID else {
                    return
                }
                isPreparingTranscriptionEngine = false
                isTranscriptionEngineReady = true
                transcriptionEnginePreparationError = nil
            } catch {
                guard !(error is CancellationError),
                      transcriptionEnginePreparationOperationID == operationID else {
                    return
                }
                isPreparingTranscriptionEngine = false
                isTranscriptionEngineReady = false
                transcriptionEnginePreparationError = error.localizedDescription
            }
        }
    }

    func makeLiveTranscriptionSession(
        inputFormat: AVAudioFormat
    ) throws -> any LiveTranscriptionSession {
        try transcriptionEngineCoordinator.makeSession(inputFormat: inputFormat)
    }

    func transcribeAudio(at url: URL) async throws -> String {
        if !isTranscriptionEngineReady {
            try await transcriptionEngineCoordinator.prepare(
                configuration: TranscriptionConfiguration(
                    engine: transcriptionEngine,
                    language: transcriptionLanguage
                )
            )
            isTranscriptionEngineReady = true
            isPreparingTranscriptionEngine = false
        }
        return try await transcriptionEngineCoordinator.transcribe(audioURL: url)
    }

    func beginStreamingTranscriptionForReview(initialTranscript: String) {
        transcriptionForReviewTask?.cancel()
        transcriptionForReviewTask = nil
        liveTranscript = initialTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        isTranscribingForReview = true
    }

    func completeStreamingTranscriptionForReview(_ transcript: String) {
        transcriptionForReviewTask?.cancel()
        transcriptionForReviewTask = nil
        let normalized = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalized.isEmpty {
            liveTranscript = normalized
        }
        isTranscribingForReview = false
    }

    func failStreamingTranscriptionForReview() {
        transcriptionForReviewTask?.cancel()
        transcriptionForReviewTask = nil
        isTranscribingForReview = false
    }

    func beginTranscriptionForReview(url: URL, initialTranscript: String = "") {
        transcriptionForReviewTask?.cancel()
        isTranscribingForReview = true
        liveTranscript = initialTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        transcriptionForReviewTask = Task { [weak self] in
            guard let self else { return }
            do {
                let text = try await transcribeAudio(at: url)
                guard !Task.isCancelled else { return }
                self.liveTranscript = text.trimmingCharacters(in: .whitespacesAndNewlines)
                self.isTranscribingForReview = false
            } catch {
                guard !Task.isCancelled else { return }
                self.isTranscribingForReview = false
            }
            self.transcriptionForReviewTask = nil
        }
    }

    func cancelTranscriptionForReview() {
        transcriptionForReviewTask?.cancel()
        transcriptionForReviewTask = nil
        isTranscribingForReview = false
        liveTranscript = ""
    }

    private static func socialDotIsMoreRecent(_ lhs: APISocialDot, _ rhs: APISocialDot) -> Bool {
        let lhsUpdatedAt = parseSocialDotUpdatedAt(lhs.updatedAt) ?? .distantPast
        let rhsUpdatedAt = parseSocialDotUpdatedAt(rhs.updatedAt) ?? .distantPast
        if lhsUpdatedAt != rhsUpdatedAt {
            return lhsUpdatedAt > rhsUpdatedAt
        }

        let lhsLocalDate = parseSocialDotLocalDate(lhs.localDate) ?? .distantPast
        let rhsLocalDate = parseSocialDotLocalDate(rhs.localDate) ?? .distantPast
        if lhsLocalDate != rhsLocalDate {
            return lhsLocalDate > rhsLocalDate
        }

        return lhs.id < rhs.id
    }

    private static func parseSocialDotLocalDate(_ rawValue: String?) -> Date? {
        guard let rawValue, !rawValue.isEmpty else { return nil }
        return DateFormatter.localDate.date(from: rawValue)
    }

    private static func parseSocialDotUpdatedAt(_ rawValue: String?) -> Date? {
        guard let rawValue, !rawValue.isEmpty else { return nil }
        if let parsed = socialDotUpdatedAtWithFractional.date(from: rawValue) {
            return parsed
        }
        return socialDotUpdatedAtBasic.date(from: rawValue)
    }

    private static let socialDotUpdatedAtWithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let socialDotUpdatedAtBasic: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static func insightRuntimeSummary(from metadata: [String: JSONValue]?) -> String? {
        guard let metadata else { return nil }
        guard let provider = stringValue(metadata["insight_provider"]) else { return nil }

        if let latency = intValue(metadata["insight_latency_ms"]) {
            if let model = stringValue(metadata["insight_model"]), !model.isEmpty {
                return "\(provider) • \(latency)ms • \(model)"
            }
            return "\(provider) • \(latency)ms"
        }

        return provider
    }

    private static func stringValue(_ value: JSONValue?) -> String? {
        guard let value else { return nil }
        if case .string(let output) = value {
            return output
        }
        return nil
    }

    private static func intValue(_ value: JSONValue?) -> Int? {
        guard let value else { return nil }
        switch value {
        case .int(let output):
            return output
        case .double(let output):
            return Int(output.rounded())
        default:
            return nil
        }
    }

    private static func sanitizedMoodTags(_ tags: [String]) -> [String] {
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if canonical.isEmpty || ordered.contains(canonical) {
                continue
            }
            ordered.append(canonical)
            if ordered.count >= 4 {
                break
            }
        }
        return ordered
    }

    private func publishSocialDot(
        localDate: String,
        insight: APIInsight
    ) async throws {
        guard let profile = currentSocialProfile else { return }
        try await socialClient.publishLocalDot(
            for: profile,
            publication: SocialDotPublication(
                localDate: localDate,
                moodScore: insight.moodScore,
                moodTags: insight.moodTags,
                dotColor: EmotionColorMixer.mixedHex(for: insight.moodTags)
            )
        )
    }

    @discardableResult
    func updateEntryTags(entryID: String, moodTags: [String]) async -> APIEntry? {
        guard !userID.isEmpty else {
            errorMessage = "Missing local user identity."
            return nil
        }

        let normalizedTags = Self.sanitizedMoodTags(moodTags)
        guard !normalizedTags.isEmpty else {
            errorMessage = "Select at least one tag."
            return nil
        }

        do {
            let updateResult = try localStore.updateEntryTags(
                for: userID,
                entryID: entryID,
                moodTags: normalizedTags
            )
            await iCloudSyncService.queueEntryUpsert(userID: userID, entryID: entryID)

            if updateResult.wasSharedToSocial,
               let insight = updateResult.updatedEntry.insight {
                do {
                    try await publishSocialDot(
                        localDate: updateResult.updatedEntry.localDate,
                        insight: insight
                    )
                } catch {
                    errorMessage = "Tags updated locally, but iCloud social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }
            await iCloudSyncService.syncNow(userID: userID, reason: "update_entry_tags")

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = describeError(error)
            return nil
        }
    }

    @discardableResult
    func updateEntryTitle(entryID: String, title: String) async -> APIEntry? {
        guard !userID.isEmpty else {
            errorMessage = "Missing local user identity."
            return nil
        }

        do {
            let updatedEntry = try localStore.updateEntryTitle(
                for: userID,
                entryID: entryID,
                title: title
            )
            await iCloudSyncService.queueEntryUpsert(userID: userID, entryID: entryID)
            await refreshEntries()
            await iCloudSyncService.syncNow(userID: userID, reason: "update_entry_title")
            return entries.first(where: { $0.id == entryID }) ?? updatedEntry
        } catch {
            errorMessage = error.localizedDescription
            return nil
        }
    }

    @discardableResult
    func retranscribeEntry(entryID: String) async -> APIEntry? {
        guard !userID.isEmpty else {
            errorMessage = "Missing local user identity."
            return nil
        }

        let baseEntry = entries.first(where: { $0.id == entryID }) ?? localStore.listEntries(for: userID).first(where: { $0.id == entryID })
        guard let baseEntry else {
            errorMessage = "Entry not found."
            return nil
        }

        do {
            let audioURL = try localStore.audioURL(for: userID, entryID: entryID)
            let durationSeconds = max(1, min(baseEntry.durationSeconds, 300))
            let selectedTranscript = try await transcribeAudio(at: audioURL)
            let (transcript, insight, generatedTitle) = await LocalReflectionAnalyzer.analyze(
                audioURL: audioURL,
                durationSeconds: durationSeconds,
                useLiquidInsights: liquidModelPrepared,
                overrideTranscriptText: selectedTranscript,
                transcriptionConfiguration: TranscriptionConfiguration(
                    engine: transcriptionEngine,
                    language: transcriptionLanguage
                )
            )
            guard let transcript else {
                errorMessage = "Could not transcribe this audio note. Try again."
                return nil
            }
            let replacementTitle = baseEntry.displayTitle == "Entry" ? generatedTitle : nil

            let updateResult = try localStore.updateEntryAnalysis(
                for: userID,
                entryID: entryID,
                transcript: transcript,
                insight: insight,
                title: replacementTitle
            )
            await iCloudSyncService.queueEntryUpsert(userID: userID, entryID: entryID)

            if updateResult.wasSharedToSocial,
               let updatedInsight = updateResult.updatedEntry.insight {
                do {
                    try await publishSocialDot(
                        localDate: updateResult.updatedEntry.localDate,
                        insight: updatedInsight
                    )
                } catch {
                    errorMessage = "Retranscribed locally, but iCloud social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }
            await iCloudSyncService.syncNow(userID: userID, reason: "retranscribe_entry")

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = describeError(error)
            return nil
        }
    }

    func deleteEntry(entryID: String) async {
        guard !userID.isEmpty else {
            errorMessage = "Missing local user identity."
            return
        }

        do {
            let result = try localStore.deleteEntry(for: userID, entryID: entryID)
            await iCloudSyncService.queueEntryDelete(userID: userID, entryID: entryID)
            if let profile = currentSocialProfile {
                do {
                    if let replacement = result.replacementSharedEntryForDate,
                       let replacementInsight = replacement.insight {
                        try await publishSocialDot(
                            localDate: result.deletedEntry.localDate,
                            insight: replacementInsight
                        )
                    } else {
                        try await socialClient.deleteLocalDot(
                            for: profile,
                            localDate: result.deletedEntry.localDate
                        )
                    }
                } catch {
                    errorMessage = "Entry deleted locally, but iCloud social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            await refreshSocialDots()
            await iCloudSyncService.syncNow(userID: userID, reason: "delete_entry")
        } catch {
            errorMessage = describeError(error)
        }
    }

    func audioDurationForDraft(_ url: URL) -> Int {
        let asset = AVURLAsset(url: url)
        let seconds = Int(CMTimeGetSeconds(asset.duration))
        if seconds > 0 {
            return seconds
        }
        return 60
    }

    func fetchAudio(entryID: String) async throws -> Data {
        guard !userID.isEmpty else {
            throw AppModelError.missingLocalUserIdentity
        }
        return try localStore.audioData(for: userID, entryID: entryID)
    }

    func createInvite() async {
        guard let profile = currentSocialProfile else {
            errorMessage = "Sign in required"
            return
        }
        do {
            let invite = try await socialClient.createInvite(for: profile)
            inviteURL = invite.inviteUrl
            inviteToken = invite.inviteToken
        } catch {
            errorMessage = describeError(error)
        }
    }

    func acceptInvite(link: String) async {
        guard currentSocialProfile != nil else {
            errorMessage = "Sign in required"
            return
        }
        do {
            try await socialClient.acceptInvite(from: cleanInviteLink(link))
            await refreshSocialDots()
        } catch {
            errorMessage = describeError(error)
        }
    }

    private func cleanInviteLink(_ value: String) -> String {
        value
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`<> \n\r\t"))
            .removingPercentEncoding ?? value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func describeError(_ error: Error) -> String {
        error.localizedDescription
    }

    func configureDailyReminder() async {
        do {
            let granted = await NotificationScheduler.requestPermission()
            notificationEnabled = granted

            if granted {
                let selectedDays = ReminderWeekday.ordered.filter { reminderWeekdays.contains($0.rawValue) }
                if selectedDays.isEmpty {
                    try await NotificationScheduler.scheduleWeekly(reminders: [])
                    reminderStatus = "No reminder days selected."
                } else {
                    let reminders = selectedDays.map { day in
                        NotificationScheduler.WeeklyReminder(
                            weekday: day,
                            time: reminderTimesByWeekday[day.rawValue] ?? dailyCheckin
                        )
                    }
                    try await NotificationScheduler.scheduleWeekly(reminders: reminders)
                    let preview = reminders
                        .prefix(3)
                        .map { "\($0.weekday.shortTitle) \(DateFormatter.clock.string(from: $0.time))" }
                        .joined(separator: ", ")
                    let suffix = reminders.count > 3 ? ", ..." : ""
                    reminderStatus = "Reminders set: \(preview)\(suffix)"
                }
            } else {
                reminderStatus = "Notifications are disabled in iOS settings."
            }

            await saveProfile()
            persistState()
        } catch {
            errorMessage = "Could not set reminder: \(error.localizedDescription)"
        }
    }

    func isReminderWeekdaySelected(_ day: ReminderWeekday) -> Bool {
        reminderWeekdays.contains(day.rawValue)
    }

    func toggleReminderWeekday(_ day: ReminderWeekday) {
        if reminderWeekdays.contains(day.rawValue) {
            reminderWeekdays.remove(day.rawValue)
        } else {
            reminderWeekdays.insert(day.rawValue)
        }
        if reminderTimesByWeekday[day.rawValue] == nil {
            reminderTimesByWeekday[day.rawValue] = dailyCheckin
        }
        persistState()
    }

    func reminderTime(for day: ReminderWeekday) -> Date {
        reminderTimesByWeekday[day.rawValue] ?? dailyCheckin
    }

    func setReminderTime(_ value: Date, for day: ReminderWeekday) {
        reminderTimesByWeekday[day.rawValue] = value
        persistState()
    }

    func selectedReminderDays() -> [ReminderWeekday] {
        ReminderWeekday.ordered.filter { reminderWeekdays.contains($0.rawValue) }
    }

    private func configureICloudSyncForCurrentUser() async {
        guard !userID.isEmpty else {
            iCloudSyncStatus = iCloudSyncEnabled ? .unavailable : .disabled
            return
        }

        await iCloudSyncService.setEnabled(iCloudSyncEnabled, userID: userID)
        guard iCloudSyncEnabled else { return }

        enqueueBootstrapICloudBackfillIfNeeded(for: userID)
        await iCloudSyncService.syncNow(userID: userID, reason: "startup")
    }

    private func enqueueBootstrapICloudBackfillIfNeeded(for userID: String) {
        guard !userID.isEmpty else { return }
        let defaults = UserDefaults.standard
        var syncedUserIDs = Set(defaults.stringArray(forKey: Keys.iCloudBootstrapSyncedUsers) ?? [])
        if syncedUserIDs.contains(userID) {
            return
        }

        Task {
            await iCloudSyncService.queueBootstrapBackfill(userID: userID)
            await iCloudSyncService.syncNow(userID: userID, reason: "bootstrap_backfill")
        }

        syncedUserIDs.insert(userID)
        defaults.set(Array(syncedUserIDs).sorted(), forKey: Keys.iCloudBootstrapSyncedUsers)
    }

    private enum HealthSnapshotTimeoutError: Error {
        case timedOut
    }

    private func fetchHealthSnapshotWithTimeout(
        at timestamp: Date,
        timeoutSeconds: Double
    ) async throws -> EntryHealthSnapshot? {
        try await withThrowingTaskGroup(of: EntryHealthSnapshot?.self) { group in
            group.addTask { [healthKitManager] in
                try await healthKitManager.fetchSnapshot(at: timestamp)
            }
            group.addTask {
                let capped = max(0.2, timeoutSeconds)
                try await Task.sleep(nanoseconds: UInt64((capped * 1_000_000_000).rounded()))
                throw HealthSnapshotTimeoutError.timedOut
            }

            let firstCompleted = try await group.next() ?? nil
            group.cancelAll()
            return firstCompleted
        }
    }

    private static func defaultReminderTimes(base: Date) -> [Int: Date] {
        ReminderWeekday.ordered.reduce(into: [Int: Date]()) { partialResult, day in
            partialResult[day.rawValue] = base
        }
    }
}
