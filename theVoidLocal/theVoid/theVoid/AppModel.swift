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

// MARK: - App model

@MainActor
final class AppModel: ObservableObject {
    private static let socialHistoryLimit = 300

    @Published var apiBaseURL: String = BackendClient.defaultBaseURLString
    @Published var apiConnectionStatus: String?
    @Published var sessionToken: String?
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

    @Published var whisperModelPrepared: Bool = false
    @Published var isPreparingWhisperModel: Bool = false
    @Published var whisperModelPreparationError: String?
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
    @Published var needsSessionReauthentication: Bool = false
    @Published var showsSessionReauthenticationPrompt: Bool = false
    @Published var sessionReauthenticationMessage: String?

    private let client = BackendClient()
    private let draftStore = DraftStore()
    private let localStore = LocalJournalStore()
    private let healthKitManager = HealthKitManager()
    private let iCloudSyncService: ICloudSyncService
    private var liquidModelPreparationTask: Task<Void, Never>?
    private var liquidModelPreparationOperationID: UUID?
    private var whisperModelPreparationTask: Task<Void, Never>?
    private var transcriptionForReviewTask: Task<Void, Never>?

    private enum Keys {
        static let apiBaseURL = "thevoid.apiBaseURL"
        static let sessionToken = "thevoid.sessionToken"
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
        static let whisperModelPrepared = "thevoid.whisperModelPrepared"
        static let healthIntegrationEnabled = "thevoid.healthkit.enabled"
        static let iCloudSyncEnabled = "thevoid.icloud.sync.enabled"
        static let iCloudBootstrapSyncedUsers = "thevoid.icloud.bootstrap.synced_users"
    }

    var needsOnboarding: Bool {
        sessionToken == nil || !onboardingComplete
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

    init() {
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
        if sessionToken != nil {
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
        if !whisperModelPrepared {
            prepareWhisperModelIfNeeded()
        }
    }

    func loadPersistedState() {
        let defaults = UserDefaults.standard
        let persistedBaseURL = defaults.string(forKey: Keys.apiBaseURL) ?? BackendClient.defaultBaseURLString
        if (try? client.updateBaseURL(persistedBaseURL)) != nil {
            apiBaseURL = client.baseURLString
        } else {
            apiBaseURL = BackendClient.defaultBaseURLString
            try? client.updateBaseURL(apiBaseURL)
        }
        sessionToken = defaults.string(forKey: Keys.sessionToken)
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
        whisperModelPrepared = defaults.bool(forKey: Keys.whisperModelPrepared)
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
        defaults.set(apiBaseURL, forKey: Keys.apiBaseURL)
        defaults.set(sessionToken, forKey: Keys.sessionToken)
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
    }

    func signIn(identityToken: String, nonce: String? = nil, suggestedName: String?) async {
        do {
            let session = try await client.signInWithApple(
                identityToken: identityToken,
                nonce: nonce,
                displayName: suggestedName,
                dailyCheckinTimeLocal: DateFormatter.hhmm.string(from: dailyCheckin),
                timezone: timezone
            )

            sessionToken = session.accessToken
            userID = session.user.id
            displayName = session.user.displayName ?? ""
            anonymousHandle = session.user.anonymousHandle
            notificationEnabled = session.user.notificationEnabled
            timezone = session.user.timezone
            needsSessionReauthentication = false
            showsSessionReauthenticationPrompt = false
            sessionReauthenticationMessage = nil

            if let parsed = DateFormatter.hhmm.date(from: session.user.dailyCheckinTimeLocal) {
                dailyCheckin = parsed
            }

            persistState()
            await configureICloudSyncForCurrentUser()
            await refreshAll()
        } catch {
            errorMessage = "\(error.localizedDescription)\nAPI: \(apiBaseURL)"
        }
    }

    func signInDev(identityToken: String?) async {
        let trimmed = identityToken?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let baseToken = trimmed.isEmpty ? UUID().uuidString : trimmed
        let token = baseToken.hasPrefix("dev-") ? baseToken : "dev-\(baseToken)"
        await signIn(identityToken: token, nonce: nil, suggestedName: nil)
    }

    func completeOnboarding() {
        onboardingComplete = true
        persistState()
    }

    func saveProfile() async {
        guard let sessionToken else { return }
        do {
            let profile = try await client.updateProfile(
                token: sessionToken,
                displayName: displayName.isEmpty ? nil : displayName,
                dailyCheckinTimeLocal: DateFormatter.hhmm.string(from: dailyCheckin),
                timezone: timezone,
                notificationEnabled: notificationEnabled
            )

            displayName = profile.displayName ?? ""
            anonymousHandle = profile.anonymousHandle
            timezone = profile.timezone
            notificationEnabled = profile.notificationEnabled
            if let parsed = DateFormatter.hhmm.date(from: profile.dailyCheckinTimeLocal) {
                dailyCheckin = parsed
            }
            persistState()
        } catch {
            errorMessage = describeRemoteError(
                error,
                includeAPIBaseURL: true,
                authenticationFailureMessage: sessionRefreshMessage(for: "profile updates")
            )
        }
    }

    func signOut() {
        liquidModelPreparationTask?.cancel()
        liquidModelPreparationTask = nil
        liquidModelPreparationOperationID = nil
        showsLiquidModelPreparationScreen = false
        isPreparingLiquidModel = false
        liquidModelPreparationProgress = 0
        liquidModelPreparationStatus = "Preparing on-device AI model..."
        liquidModelPreparationError = nil
        sessionToken = nil
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
        needsSessionReauthentication = false
        showsSessionReauthenticationPrompt = false
        sessionReauthenticationMessage = nil
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
        guard let sessionToken else { return }
        do {
            let envelope = try await client.fetchSocialDots(
                token: sessionToken,
                history: true,
                limit: Self.socialHistoryLimit
            )
            socialDots = envelope.dots.sorted(by: Self.socialDotIsMoreRecent)
        } catch is CancellationError {
            // Pull-to-refresh and view transitions can cancel in-flight tasks; keep last good state.
            return
        } catch {
            errorMessage = describeRemoteError(
                error,
                authenticationFailureMessage: sessionRefreshMessage(for: "social features")
            )
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
        guard let sessionToken else {
            errorMessage = "Sign in required"
            return
        }
        guard !userID.isEmpty else {
            errorMessage = "Missing user identity. Sign in again."
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
                overrideTranscriptText: overrideTranscript
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
                    try await client.publishLocalDot(
                        token: sessionToken,
                        localDate: localDate,
                        moodScore: insight.moodScore,
                        moodTags: insight.moodTags,
                        dotColor: EmotionColorMixer.mixedHex(for: insight.moodTags)
                    )
                } catch {
                    errorMessage = describeRemoteError(
                        error,
                        authenticationFailureMessage: "Saved locally, but the backend session for \(apiBaseURL) is no longer valid. Reconnect with Sign in with Apple to restore social sync."
                    )
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
            errorMessage = describeRemoteError(error)
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
        guard sessionToken != nil, onboardingComplete else { return }
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

    // MARK: - WhisperKit model

    func prepareWhisperModelIfNeeded() {
        guard !isPreparingWhisperModel, !whisperModelPrepared else { return }
        isPreparingWhisperModel = true
        whisperModelPreparationError = nil
        whisperModelPreparationTask?.cancel()
        whisperModelPreparationTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await WhisperTranscriptionRuntime.shared.prepare()
                await MainActor.run {
                    self.isPreparingWhisperModel = false
                    self.setWhisperModelPrepared(true)
                    self.whisperModelPreparationError = nil
                }
            } catch {
                if error is CancellationError { return }
                await MainActor.run {
                    self.isPreparingWhisperModel = false
                    self.whisperModelPreparationError = error.localizedDescription
                }
            }
        }
    }

    func cancelWhisperModelPreparation() {
        whisperModelPreparationTask?.cancel()
        whisperModelPreparationTask = nil
        isPreparingWhisperModel = false
        whisperModelPreparationError = nil
    }

    func beginTranscriptionForReview(url: URL) {
        transcriptionForReviewTask?.cancel()
        isTranscribingForReview = true
        liveTranscript = ""
        transcriptionForReviewTask = Task { [weak self] in
            guard let self else { return }
            do {
                let text = try await WhisperTranscriptionRuntime.shared.transcribe(audioURL: url)
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

    private func setWhisperModelPrepared(_ prepared: Bool) {
        whisperModelPrepared = prepared
        UserDefaults.standard.set(prepared, forKey: Keys.whisperModelPrepared)
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

            if let sessionToken,
               updateResult.wasSharedToSocial,
               let insight = updateResult.updatedEntry.insight {
                do {
                    try await client.publishLocalDot(
                        token: sessionToken,
                        localDate: updateResult.updatedEntry.localDate,
                        moodScore: insight.moodScore,
                        moodTags: insight.moodTags,
                        dotColor: EmotionColorMixer.mixedHex(for: insight.moodTags)
                    )
                } catch {
                    errorMessage = describeRemoteError(
                        error,
                        authenticationFailureMessage: "Tags updated locally, but the backend session for \(apiBaseURL) is no longer valid. Reconnect with Sign in with Apple to restore social sync."
                    )
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }
            await iCloudSyncService.syncNow(userID: userID, reason: "update_entry_tags")

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = describeRemoteError(error)
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
            let (transcript, insight, generatedTitle) = await LocalReflectionAnalyzer.analyze(
                audioURL: audioURL,
                durationSeconds: durationSeconds,
                useLiquidInsights: liquidModelPrepared
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

            if let sessionToken,
               updateResult.wasSharedToSocial,
               let updatedInsight = updateResult.updatedEntry.insight {
                do {
                    try await client.publishLocalDot(
                        token: sessionToken,
                        localDate: updateResult.updatedEntry.localDate,
                        moodScore: updatedInsight.moodScore,
                        moodTags: updatedInsight.moodTags,
                        dotColor: EmotionColorMixer.mixedHex(for: updatedInsight.moodTags)
                    )
                } catch {
                    errorMessage = describeRemoteError(
                        error,
                        authenticationFailureMessage: "Retranscribed locally, but the backend session for \(apiBaseURL) is no longer valid. Reconnect with Sign in with Apple to restore social sync."
                    )
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }
            await iCloudSyncService.syncNow(userID: userID, reason: "retranscribe_entry")

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = describeRemoteError(error)
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
            if let sessionToken {
                do {
                    if let replacement = result.replacementSharedEntryForDate,
                       let replacementInsight = replacement.insight {
                        let replacementColor = EmotionColorMixer.mixedHex(for: replacementInsight.moodTags)
                        try await client.publishLocalDot(
                            token: sessionToken,
                            localDate: result.deletedEntry.localDate,
                            moodScore: replacementInsight.moodScore,
                            moodTags: replacementInsight.moodTags,
                            dotColor: replacementColor
                        )
                    } else {
                        try await client.deleteLocalDot(
                            token: sessionToken,
                            localDate: result.deletedEntry.localDate
                        )
                    }
                } catch {
                    errorMessage = describeRemoteError(
                        error,
                        authenticationFailureMessage: "Entry deleted locally, but the backend session for \(apiBaseURL) is no longer valid. Reconnect with Sign in with Apple to restore social sync."
                    )
                }
            }

            await refreshEntries()
            await refreshSocialDots()
            await iCloudSyncService.syncNow(userID: userID, reason: "delete_entry")
        } catch {
            errorMessage = describeRemoteError(error)
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
            throw APIError.server(401, "Missing local user identity")
        }
        return try localStore.audioData(for: userID, entryID: entryID)
    }

    func createInvite() async {
        guard let sessionToken else { return }
        do {
            let invite = try await client.createInvite(token: sessionToken)
            inviteURL = invite.inviteUrl
            inviteToken = invite.inviteToken
        } catch {
            errorMessage = describeRemoteError(
                error,
                includeAPIBaseURL: true,
                authenticationFailureMessage: sessionRefreshMessage(for: "social features")
            )
        }
    }

    func acceptInvite(token: String) async {
        guard let sessionToken else { return }
        let inviteToken = normalizedInviteToken(from: token)
        if inviteToken.isEmpty {
            errorMessage = "Paste a valid invite token or invite link."
            return
        }
        do {
            try await client.acceptInvite(token: sessionToken, inviteToken: inviteToken)
            await refreshSocialDots()
        } catch {
            errorMessage = describeRemoteError(
                error,
                includeAPIBaseURL: true,
                authenticationFailureMessage: sessionRefreshMessage(for: "social features")
            )
        }
    }

    @discardableResult
    func submitFeedback(kind: String, message: String) async -> Bool {
        guard let sessionToken else {
            errorMessage = "Sign in required"
            return false
        }

        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            errorMessage = "Please enter your idea or bug report."
            return false
        }

        do {
            try await client.submitFeedback(token: sessionToken, kind: kind, message: trimmed)
            return true
        } catch {
            errorMessage = describeRemoteError(
                error,
                includeAPIBaseURL: true,
                authenticationFailureMessage: sessionRefreshMessage(for: "feedback")
            )
            return false
        }
    }

    private func normalizedInviteToken(from rawValue: String) -> String {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return ""
        }

        let stripped = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`<>"))

        if let components = URLComponents(string: stripped),
           let token = components.queryItems?.first(where: { $0.name == "token" })?.value {
            return cleanInviteToken(token)
        }

        if let tokenRange = stripped.range(of: "token=") {
            var candidate = String(stripped[tokenRange.upperBound...])
            if let cut = candidate.firstIndex(where: { $0 == "&" || $0.isWhitespace }) {
                candidate = String(candidate[..<cut])
            }
            return cleanInviteToken(candidate)
        }

        return cleanInviteToken(stripped)
    }

    private func cleanInviteToken(_ value: String) -> String {
        let decoded = value.removingPercentEncoding ?? value
        return decoded.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`<> \n\r\t"))
    }

    func applyAPIBaseURL() {
        let previousBaseURL = client.baseURLString
        do {
            try client.updateBaseURL(apiBaseURL)
            apiBaseURL = client.baseURLString
            if sessionToken != nil, apiBaseURL != previousBaseURL {
                presentSessionReauthenticationPrompt(
                    message: "You changed the backend to \(apiBaseURL). Sign in again so social features use a valid session on this server."
                )
                apiConnectionStatus = "API URL saved. Reconnect session."
            } else {
                apiConnectionStatus = "API URL saved"
            }
            persistState()
        } catch {
            errorMessage = "Invalid API base URL"
        }
    }

    func testAPIConnection() async {
        do {
            let health = try await client.healthCheck()
            apiConnectionStatus = "Connected (\(health.status))"
        } catch {
            apiConnectionStatus = "Connection failed"
            errorMessage = describeRemoteError(error, includeAPIBaseURL: true)
        }
    }

    func dismissSessionReauthenticationPrompt() {
        showsSessionReauthenticationPrompt = false
    }

#if DEBUG
    func simulateExpiredSessionForTesting() async {
        guard sessionToken != nil else {
            errorMessage = "Sign in required"
            return
        }

        sessionToken = "expired-test-\(UUID().uuidString)"
        persistState()
        await refreshSocialDots()
    }
#endif

    private func sessionRefreshMessage(for featureName: String) -> String {
        "Your backend session is no longer valid for \(apiBaseURL). Reconnect with Sign in with Apple to restore \(featureName)."
    }

    private func presentSessionReauthenticationPrompt(message: String) {
        needsSessionReauthentication = true
        showsSessionReauthenticationPrompt = true
        sessionReauthenticationMessage = message
    }

    private func describeRemoteError(
        _ error: Error,
        includeAPIBaseURL: Bool = false,
        authenticationFailureMessage: String? = nil
    ) -> String? {
        if let apiError = error as? APIError, apiError.isAuthenticationFailure {
            presentSessionReauthenticationPrompt(
                message: authenticationFailureMessage ?? sessionRefreshMessage(for: "social features")
            )
            return nil
        }

        let message = error.localizedDescription
        if includeAPIBaseURL {
            return "\(message)\nAPI: \(apiBaseURL)"
        }
        return message
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
