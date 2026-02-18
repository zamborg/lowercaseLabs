//
//  ContentView.swift
//  theVoid
//
//  Created by zubin aysola on 2/16/26.
//

import AuthenticationServices
import AVFoundation
import CryptoKit
import SwiftUI
import UIKit
import UserNotifications

// MARK: - DTOs

struct APIUserProfile: Codable {
    let id: String
    let displayName: String?
    let anonymousHandle: String
    let dailyCheckinTimeLocal: String
    let timezone: String
    let notificationEnabled: Bool
}

struct APIAuthSession: Codable {
    let accessToken: String
    let tokenType: String
    let user: APIUserProfile
}

struct APIEntryCreate: Codable {
    let entryId: String
    let uploadUrl: String
    let uploadMethod: String
    let objectKey: String
    let expiresAt: String
}

enum JSONValue: Codable, Hashable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

struct APITranscript: Codable, Hashable {
    let text: String
    let providerMetadata: [String: JSONValue]?
}

struct APIInsight: Codable, Hashable {
    let moodScore: Double
    let moodTags: [String]
    let summary: String
    let themes: [String]
    let signals: [String: Double]
    let safetyFlags: [String: JSONValue]?
}

struct APIEntry: Codable, Identifiable, Hashable {
    let id: String
    let localDate: String
    let durationSeconds: Int
    let status: String
    let createdAt: String
    let transcript: APITranscript?
    let insight: APIInsight?
}

struct APISocialDot: Codable, Identifiable, Hashable {
    var id: String { userId }
    let userId: String
    let dotColor: String
    let label: String?
    let isRevealed: Bool
    let hasEntry: Bool
}

struct APISocialDotsEnvelope: Codable {
    let localDate: String
    let dots: [APISocialDot]
}

struct APIInvite: Codable {
    let inviteToken: String
    let inviteUrl: String
    let expiresAt: String
}

struct APIMessage: Codable {
    let message: String
}

enum AppleNonce {
    static func random(length: Int = 32) -> String {
        let charset: [Character] = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")
        var result = ""
        result.reserveCapacity(length)

        while result.count < length {
            let randomByte = UInt8.random(in: 0 ... 255)
            if randomByte < charset.count {
                result.append(charset[Int(randomByte)])
            }
        }
        return result
    }

    static func sha256(_ value: String) -> String {
        let inputData = Data(value.utf8)
        let hashed = SHA256.hash(data: inputData)
        return hashed.compactMap { String(format: "%02x", $0) }.joined()
    }
}

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

enum RevealModeOption: String, CaseIterable, Identifiable, Codable {
    case anonymous = "anonymous"
    case revealedToFriends = "revealed_to_friends"
    case revealedToSpecific = "revealed_to_specific"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .anonymous:
            return "Anonymous"
        case .revealedToFriends:
            return "Friends"
        case .revealedToSpecific:
            return "Selected"
        }
    }
}

// MARK: - Networking

enum APIError: LocalizedError {
    case invalidURL
    case transport(String)
    case server(Int, String)
    case decoding

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API URL."
        case .transport(let message):
            return message
        case .server(_, let message):
            return message
        case .decoding:
            return "Could not decode server response."
        }
    }
}

final class BackendClient {
    static let localBaseURLString = "http://127.0.0.1:8080"
    static let productionBaseURLString = "https://thevoid.fly.dev"
    static let defaultBaseURLString = BackendClient.localBaseURLString

    private var baseURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(baseURL: URL = URL(string: BackendClient.defaultBaseURLString)!) {
        self.baseURL = baseURL
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
        self.encoder.keyEncodingStrategy = .convertToSnakeCase
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    var baseURLString: String {
        baseURL.absoluteString
    }

    struct AuthPayload: Encodable {
        let identityToken: String
        let nonce: String?
        let displayName: String?
        let dailyCheckinTimeLocal: String
        let timezone: String
    }

    struct EntryCreatePayload: Encodable {
        let localDate: String
        let durationSeconds: Int
    }

    struct CompleteUploadPayload: Encodable {
        let contentType: String?
    }

    struct UpdateProfilePayload: Encodable {
        let displayName: String?
        let dailyCheckinTimeLocal: String
        let timezone: String
        let notificationEnabled: Bool
    }

    struct UpdateSocialPresencePayload: Encodable {
        let revealMode: String
        let revealFriendIds: [String]
        let displayNameOverride: String?
    }

    struct InvitePayload: Encodable {
        let expiresInDays: Int
        let maxUses: Int
    }

    struct AcceptInvitePayload: Encodable {
        let token: String
    }

    struct HealthResponse: Decodable {
        let status: String
        let time: String
    }

    func updateBaseURL(_ value: String) throws {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              (scheme == "http" || scheme == "https"),
              url.host != nil else {
            throw APIError.invalidURL
        }
        baseURL = url
    }

    private func buildRequest(
        url: URL,
        method: String,
        token: String? = nil,
        body: Data? = nil,
        contentType: String = "application/json"
    ) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = body
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    private func decodeError(_ data: Data, statusCode: Int) -> APIError {
        if let message = try? decoder.decode(APIMessage.self, from: data).message {
            return .server(statusCode, message)
        }
        let fallback = String(data: data, encoding: .utf8) ?? "Request failed"
        return .server(statusCode, fallback)
    }

    private func send<T: Decodable>(_ request: URLRequest, decode: T.Type) async throws -> T {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }

        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding
        }
    }

    private func sendVoid(_ request: URLRequest) async throws {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }

        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }
    }

    func signInWithApple(identityToken: String, nonce: String?, displayName: String?, dailyCheckinTimeLocal: String, timezone: String) async throws -> APIAuthSession {
        let payload = AuthPayload(
            identityToken: identityToken,
            nonce: nonce,
            displayName: displayName,
            dailyCheckinTimeLocal: dailyCheckinTimeLocal,
            timezone: timezone
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/auth/apple", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", body: body)
        return try await send(request, decode: APIAuthSession.self)
    }

    func updateProfile(token: String, displayName: String?, dailyCheckinTimeLocal: String, timezone: String, notificationEnabled: Bool) async throws -> APIUserProfile {
        let payload = UpdateProfilePayload(
            displayName: displayName,
            dailyCheckinTimeLocal: dailyCheckinTimeLocal,
            timezone: timezone,
            notificationEnabled: notificationEnabled
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/me", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "PATCH", token: token, body: body)
        return try await send(request, decode: APIUserProfile.self)
    }

    func createEntry(token: String, localDate: String, durationSeconds: Int) async throws -> APIEntryCreate {
        let payload = EntryCreatePayload(localDate: localDate, durationSeconds: durationSeconds)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/entries", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        return try await send(request, decode: APIEntryCreate.self)
    }

    func uploadAudio(uploadURL: String, token: String, audioData: Data) async throws {
        guard let url = URL(string: uploadURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(
            url: url,
            method: "PUT",
            token: token,
            body: audioData,
            contentType: "application/octet-stream"
        )
        try await sendVoid(request)
    }

    func completeUpload(entryID: String, token: String) async throws {
        let payload = CompleteUploadPayload(contentType: "audio/m4a")
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/entries/\(entryID)/complete_upload", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func fetchEntries(token: String) async throws -> [APIEntry] {
        guard let url = URL(string: "/entries", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: [APIEntry].self)
    }

    func fetchEntry(entryID: String, token: String) async throws -> APIEntry {
        guard let url = URL(string: "/entries/\(entryID)", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: APIEntry.self)
    }

    func fetchAudio(entryID: String, token: String) async throws -> Data {
        guard let url = URL(string: "/entries/\(entryID)/audio", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("Invalid response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw decodeError(data, statusCode: http.statusCode)
        }
        return data
    }

    func fetchSocialDots(token: String, localDate: String) async throws -> APISocialDotsEnvelope {
        guard let url = URL(string: "/social/dots?local_date=\(localDate)", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: APISocialDotsEnvelope.self)
    }

    func updateSocialPresence(
        token: String,
        localDate: String,
        revealMode: RevealModeOption,
        revealFriendIDs: [String],
        displayNameOverride: String?
    ) async throws {
        let payload = UpdateSocialPresencePayload(
            revealMode: revealMode.rawValue,
            revealFriendIds: revealFriendIDs,
            displayNameOverride: displayNameOverride
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/social/presence/\(localDate)", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "PATCH", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func createInvite(token: String) async throws -> APIInvite {
        let payload = InvitePayload(expiresInDays: 7, maxUses: 1)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/friends/invite", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        return try await send(request, decode: APIInvite.self)
    }

    func acceptInvite(token: String, inviteToken: String) async throws {
        let payload = AcceptInvitePayload(token: inviteToken)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/friends/accept", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "POST", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func healthCheck() async throws -> HealthResponse {
        guard let url = URL(string: "/health", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET")
        return try await send(request, decode: HealthResponse.self)
    }
}

// MARK: - Local draft store

final class DraftStore {
    private let directory: URL

    init() {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.directory = root.appendingPathComponent("VoidDrafts", isDirectory: true)
        if !FileManager.default.fileExists(atPath: directory.path) {
            try? FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }

    func makeDraftURL() -> URL {
        directory.appendingPathComponent("\(UUID().uuidString).m4a")
    }

    func listDrafts() -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.creationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return urls.sorted {
            let leftDate = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let rightDate = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return leftDate > rightDate
        }
    }

    func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}

// MARK: - Recorder

@MainActor
final class RecorderEngine: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var elapsed: TimeInterval = 0
    @Published var amplitude: CGFloat = 0.05

    var onWarning: ((TimeInterval) -> Void)?
    var onAutoStop: (() -> Void)?

    private var recorder: AVAudioRecorder?
    private var meterTimer: Timer?
    private var startDate: Date?
    private var currentURL: URL?
    private var warned430 = false
    private var warned455 = false

    func recordPermissionStatus() -> AVAudioSession.RecordPermission {
        AVAudioSession.sharedInstance().recordPermission
    }

    func requestPermission() async -> Bool {
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            break
        @unknown default:
            break
        }

        return await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording(at url: URL) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 12_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]

        recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder?.isMeteringEnabled = true
        recorder?.record()

        currentURL = url
        startDate = Date()
        warned430 = false
        warned455 = false
        elapsed = 0
        amplitude = 0.05
        isRecording = true

        meterTimer?.invalidate()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    func stopRecording() -> URL? {
        meterTimer?.invalidate()
        meterTimer = nil
        recorder?.stop()
        isRecording = false
        let finishedURL = currentURL
        currentURL = nil
        return finishedURL
    }

    private func tick() {
        guard let recorder, let startDate else {
            return
        }

        recorder.updateMeters()
        let power = recorder.averagePower(forChannel: 0)
        let normalized = max(0.04, CGFloat((power + 160.0) / 160.0))

        elapsed = Date().timeIntervalSince(startDate)
        amplitude = normalized

        if elapsed >= 270, !warned430 {
            warned430 = true
            onWarning?(270)
        }

        if elapsed >= 295, !warned455 {
            warned455 = true
            onWarning?(295)
        }

        if elapsed >= 300 {
            _ = stopRecording()
            onAutoStop?()
        }
    }
}

// MARK: - Audio playback

@MainActor
final class AudioPlaybackController: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var isLoading = false
    @Published var isReady = false
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0

    private var player: AVAudioPlayer?
    private var progressTimer: Timer?
    private var isScrubbing = false

    func load(fetchAudio: () async throws -> Data, forceReload: Bool = false) async throws {
        if isLoading {
            return
        }
        if isReady && !forceReload {
            return
        }

        isLoading = true
        defer { isLoading = false }

        let data = try await fetchAudio()
        try configurePlayer(with: data)
    }

    func togglePlayback() {
        guard let player else {
            return
        }

        if player.isPlaying {
            player.pause()
            isPlaying = false
            stopTimer()
            return
        }

        player.currentTime = currentTime
        if player.play() {
            isPlaying = true
            startTimer()
        }
    }

    func restart() {
        guard let player else {
            return
        }
        player.currentTime = 0
        currentTime = 0
        if !player.isPlaying {
            _ = player.play()
        }
        isPlaying = player.isPlaying
        startTimer()
    }

    func stop() {
        guard let player else {
            return
        }
        player.stop()
        player.currentTime = 0
        currentTime = 0
        isPlaying = false
        stopTimer()
    }

    func beginScrubbing() {
        isScrubbing = true
    }

    func scrub(to time: TimeInterval) {
        currentTime = min(max(time, 0), max(duration, 0))
    }

    func endScrubbing() {
        guard let player else {
            isScrubbing = false
            return
        }
        player.currentTime = currentTime
        isScrubbing = false
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        currentTime = duration
        stopTimer()
    }

    private func configurePlayer(with data: Data) throws {
        stopTimer()
        player?.stop()

        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [])
        try? session.setActive(true, options: .notifyOthersOnDeactivation)

        let nextPlayer = try AVAudioPlayer(data: data)
        nextPlayer.delegate = self
        nextPlayer.prepareToPlay()

        player = nextPlayer
        duration = nextPlayer.duration
        currentTime = 0
        isPlaying = false
        isReady = true
    }

    private func startTimer() {
        stopTimer()
        let timer = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    private func stopTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }

    private func tick() {
        guard let player else {
            return
        }
        if !isScrubbing {
            currentTime = player.currentTime
        }
        isPlaying = player.isPlaying
    }
}

// MARK: - Notification scheduler

enum NotificationScheduler {
    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
                continuation.resume(returning: granted)
            }
        }
    }

    static func scheduleDaily(at date: Date) async throws {
        let center = UNUserNotificationCenter.current()
        center.removePendingNotificationRequests(withIdentifiers: ["void.daily.checkin"])

        let content = UNMutableNotificationContent()
        content.title = "Step Into the Void"
        content.body = "Your one intentional check-in is ready."
        content.sound = .default

        let components = Calendar.current.dateComponents([.hour, .minute], from: date)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
        let request = UNNotificationRequest(identifier: "void.daily.checkin", content: content, trigger: trigger)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            center.add(request) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
        }
    }

    static func authorizationStatus() async -> UNAuthorizationStatus {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().getNotificationSettings { settings in
                continuation.resume(returning: settings.authorizationStatus)
            }
        }
    }
}

// MARK: - App model

@MainActor
final class AppModel: ObservableObject {
    @Published var apiBaseURL: String = BackendClient.defaultBaseURLString
    @Published var apiConnectionStatus: String?
    @Published var sessionToken: String?
    @Published var displayName: String = ""
    @Published var anonymousHandle: String = ""
    @Published var dailyCheckin: Date = Date()
    @Published var timezone: String = TimeZone.current.identifier
    @Published var notificationEnabled: Bool = false
    @Published var onboardingComplete: Bool = false

    @Published var entries: [APIEntry] = []
    @Published var socialDots: [APISocialDot] = []
    @Published var pendingDrafts: [URL] = []
    @Published var submissionState: SubmissionState = .idle
    @Published var activeEntryID: String?

    @Published var revealMode: RevealModeOption = .anonymous
    @Published var revealFriendIDs: Set<String> = []

    @Published var inviteURL: String?
    @Published var inviteToken: String?
    @Published var errorMessage: String?

    private let client = BackendClient()
    private let draftStore = DraftStore()

    private enum Keys {
        static let apiBaseURL = "thevoid.apiBaseURL"
        static let sessionToken = "thevoid.sessionToken"
        static let displayName = "thevoid.displayName"
        static let anonymousHandle = "thevoid.anonymousHandle"
        static let dailyCheckin = "thevoid.dailyCheckin"
        static let timezone = "thevoid.timezone"
        static let notificationEnabled = "thevoid.notificationEnabled"
        static let onboardingComplete = "thevoid.onboardingComplete"
    }

    var needsOnboarding: Bool {
        sessionToken == nil || !onboardingComplete
    }

    var latestTranscriptEntry: APIEntry? {
        entries.first { entry in
            guard let text = entry.transcript?.text else {
                return false
            }
            return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    init() {
        loadPersistedState()
        reloadDrafts()
        if sessionToken != nil {
            Task { await refreshAll() }
        }
    }

    func loadPersistedState() {
        let defaults = UserDefaults.standard
        apiBaseURL = defaults.string(forKey: Keys.apiBaseURL) ?? BackendClient.defaultBaseURLString
        sessionToken = defaults.string(forKey: Keys.sessionToken)
        displayName = defaults.string(forKey: Keys.displayName) ?? ""
        anonymousHandle = defaults.string(forKey: Keys.anonymousHandle) ?? ""
        timezone = defaults.string(forKey: Keys.timezone) ?? TimeZone.current.identifier
        notificationEnabled = defaults.object(forKey: Keys.notificationEnabled) as? Bool ?? false
        onboardingComplete = defaults.bool(forKey: Keys.onboardingComplete)

        if let timeString = defaults.string(forKey: Keys.dailyCheckin),
           let parsed = DateFormatter.hhmm.date(from: timeString) {
            dailyCheckin = parsed
        }

        do {
            try client.updateBaseURL(apiBaseURL)
            apiBaseURL = client.baseURLString
        } catch {
            apiBaseURL = BackendClient.defaultBaseURLString
            try? client.updateBaseURL(apiBaseURL)
        }
    }

    func persistState() {
        let defaults = UserDefaults.standard
        defaults.set(apiBaseURL, forKey: Keys.apiBaseURL)
        defaults.set(sessionToken, forKey: Keys.sessionToken)
        defaults.set(displayName, forKey: Keys.displayName)
        defaults.set(anonymousHandle, forKey: Keys.anonymousHandle)
        defaults.set(DateFormatter.hhmm.string(from: dailyCheckin), forKey: Keys.dailyCheckin)
        defaults.set(timezone, forKey: Keys.timezone)
        defaults.set(notificationEnabled, forKey: Keys.notificationEnabled)
        defaults.set(onboardingComplete, forKey: Keys.onboardingComplete)
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
            displayName = session.user.displayName ?? ""
            anonymousHandle = session.user.anonymousHandle
            notificationEnabled = session.user.notificationEnabled
            timezone = session.user.timezone

            if let parsed = DateFormatter.hhmm.date(from: session.user.dailyCheckinTimeLocal) {
                dailyCheckin = parsed
            }

            persistState()
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
            errorMessage = error.localizedDescription
        }
    }

    func signOut() {
        sessionToken = nil
        displayName = ""
        anonymousHandle = ""
        entries = []
        socialDots = []
        inviteURL = nil
        inviteToken = nil
        onboardingComplete = false
        persistState()
    }

    func refreshAll() async {
        await refreshEntries()
        await refreshSocialDots()
        reloadDrafts()
    }

    func refreshEntries() async {
        guard let sessionToken else { return }
        do {
            entries = try await client.fetchEntries(token: sessionToken)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func refreshSocialDots() async {
        guard let sessionToken else { return }
        do {
            let today = DateFormatter.localDate.string(from: Date())
            let envelope = try await client.fetchSocialDots(token: sessionToken, localDate: today)
            socialDots = envelope.dots
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func reloadDrafts() {
        pendingDrafts = draftStore.listDrafts()
    }

    func makeDraftURL() -> URL {
        draftStore.makeDraftURL()
    }

    func deleteDraft(_ url: URL) {
        draftStore.delete(url)
        reloadDrafts()
    }

    func submitDraft(url: URL, durationSeconds: Int) async {
        guard let sessionToken else {
            errorMessage = "Sign in required"
            return
        }

        do {
            submissionState = .uploading
            let normalizedDuration = max(1, min(durationSeconds, 300))
            let localDate = DateFormatter.localDate.string(from: Date())

            let createResponse = try await client.createEntry(
                token: sessionToken,
                localDate: localDate,
                durationSeconds: normalizedDuration
            )

            let audioData = try Data(contentsOf: url)
            try await client.uploadAudio(
                uploadURL: createResponse.uploadUrl,
                token: sessionToken,
                audioData: audioData
            )
            try await client.completeUpload(entryID: createResponse.entryId, token: sessionToken)

            draftStore.delete(url)
            reloadDrafts()

            submissionState = .transcribing
            activeEntryID = createResponse.entryId
            try await pollUntilReady(entryID: createResponse.entryId, token: sessionToken)

            submissionState = .insightsReady
            await refreshAll()
        } catch {
            submissionState = .failed
            errorMessage = error.localizedDescription
            reloadDrafts()
        }
    }

    private func pollUntilReady(entryID: String, token: String) async throws {
        for _ in 0..<24 {
            let entry = try await client.fetchEntry(entryID: entryID, token: token)
            if entry.status == "ready" {
                return
            }
            if entry.status == "failed" {
                throw APIError.server(500, "Insight pipeline failed")
            }
            try await Task.sleep(nanoseconds: 3_000_000_000)
        }
        throw APIError.server(408, "Timed out waiting for insights")
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
        guard let sessionToken else {
            throw APIError.server(401, "Session expired")
        }
        return try await client.fetchAudio(entryID: entryID, token: sessionToken)
    }

    func saveRevealMode(displayNameOverride: String?) async {
        guard let sessionToken else { return }
        let today = DateFormatter.localDate.string(from: Date())
        do {
            try await client.updateSocialPresence(
                token: sessionToken,
                localDate: today,
                revealMode: revealMode,
                revealFriendIDs: Array(revealFriendIDs),
                displayNameOverride: displayNameOverride?.isEmpty == true ? nil : displayNameOverride
            )
            await refreshSocialDots()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func createInvite() async {
        guard let sessionToken else { return }
        do {
            let invite = try await client.createInvite(token: sessionToken)
            inviteURL = invite.inviteUrl
            inviteToken = invite.inviteToken
        } catch {
            errorMessage = "\(error.localizedDescription)\nAPI: \(apiBaseURL)"
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
            errorMessage = "\(error.localizedDescription)\nAPI: \(apiBaseURL)"
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
        do {
            try client.updateBaseURL(apiBaseURL)
            apiBaseURL = client.baseURLString
            apiConnectionStatus = "API URL saved"
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
            errorMessage = "\(error.localizedDescription)\nAPI: \(apiBaseURL)"
        }
    }
}

// MARK: - Root view

struct ContentView: View {
    @StateObject private var model = AppModel()

    var body: some View {
        Group {
            if model.needsOnboarding {
                OnboardingView()
            } else {
                MainTabView()
            }
        }
        .environmentObject(model)
        .alert("Error", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            VoidExperienceView()
                .tabItem {
                    Label("Void", systemImage: "waveform")
                }

            JournalView()
                .tabItem {
                    Label("Journal", systemImage: "book")
                }

            SocialView()
                .tabItem {
                    Label("Social", systemImage: "circle.grid.3x3.fill")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
    }
}

// MARK: - Onboarding

struct OnboardingView: View {
    @EnvironmentObject private var model: AppModel
    
    private enum StartupBackend: String, CaseIterable, Identifiable {
        case prod
        case local

        var id: String { rawValue }

        var title: String {
            switch self {
            case .prod:
                return "Prod"
            case .local:
                return "Local"
            }
        }

        var urlString: String {
            switch self {
            case .prod:
                return BackendClient.productionBaseURLString
            case .local:
                return BackendClient.localBaseURLString
            }
        }

        static func from(urlString: String) -> StartupBackend {
            guard let host = URL(string: urlString)?.host?.lowercased() else {
                return .local
            }
            return host == "thevoid.fly.dev" ? .prod : .local
        }
    }

    @State private var micGranted = false
    @State private var notificationGranted = false
    @State private var devIdentityToken = ""
    @State private var currentAppleNonce: String?
    @State private var startupBackend: StartupBackend = .local

    private var appleSignInEnabled: Bool {
        startupBackend == .prod
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("theVoid")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                Text("One intentional check-in. Private by default.")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 12) {
                    Text("Backend")
                        .font(.headline)

                    Picker("Backend", selection: $startupBackend) {
                        ForEach(StartupBackend.allCases) { backend in
                            Text(backend.title).tag(backend)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: startupBackend) { _, newValue in
                        model.apiBaseURL = newValue.urlString
                        model.applyAPIBaseURL()
                    }

                    Text(model.apiBaseURL)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Identity")
                        .font(.headline)

                    if appleSignInEnabled {
                        SignInWithAppleButton(.signIn) { request in
                            request.requestedScopes = [.fullName]
                            let nonce = AppleNonce.random()
                            currentAppleNonce = nonce
                            request.nonce = AppleNonce.sha256(nonce)
                        } onCompletion: { result in
                            switch result {
                            case .failure(let error):
                                model.errorMessage = error.localizedDescription
                            case .success(let auth):
                                guard let credential = auth.credential as? ASAuthorizationAppleIDCredential,
                                      let tokenData = credential.identityToken,
                                      let token = String(data: tokenData, encoding: .utf8)
                                else {
                                    model.errorMessage = "Unable to read Apple identity token"
                                    return
                                }

                                let fullName = [credential.fullName?.givenName, credential.fullName?.familyName]
                                    .compactMap { $0 }
                                    .joined(separator: " ")
                                let nonce = currentAppleNonce
                                currentAppleNonce = nil
                                Task {
                                    await model.signIn(identityToken: token, nonce: nonce, suggestedName: fullName.isEmpty ? nil : fullName)
                                }
                            }
                        }
                        .signInWithAppleButtonStyle(.white)
                        .frame(height: 48)
                    } else {
                        Text("Use Dev Sign-In for local testing. Switch backend to Prod for Apple Sign-In.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    if !appleSignInEnabled {
                        TextField("Dev identity token (optional)", text: $devIdentityToken)
                            .textFieldStyle(.plain)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            )
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()

                        Button("Use Dev Sign-In") {
                            Task { await model.signInDev(identityToken: devIdentityToken) }
                        }
                        .buttonStyle(.bordered)
                    }

                    if !model.anonymousHandle.isEmpty {
                        Text("Signed in as @\(model.anonymousHandle)")
                            .font(.subheadline)
                    }
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Check-In")
                        .font(.headline)
                    DatePicker("Daily time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
                    TextField("Display name (optional)", text: $model.displayName)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                        )
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Permissions")
                        .font(.headline)

                    HStack {
                        Label("Microphone", systemImage: micGranted ? "checkmark.circle.fill" : "circle")
                        Spacer()
                        Button(micGranted ? "Granted" : "Allow") {
                            Task {
                                micGranted = await RecorderEngine().requestPermission()
                            }
                        }
                        .disabled(micGranted)
                    }

                    HStack {
                        Label("Notifications", systemImage: notificationGranted ? "checkmark.circle.fill" : "circle")
                        Spacer()
                        Button(notificationGranted ? "Granted" : "Allow") {
                            Task {
                                notificationGranted = await NotificationScheduler.requestPermission()
                                model.notificationEnabled = notificationGranted
                                if notificationGranted {
                                    try? await NotificationScheduler.scheduleDaily(at: model.dailyCheckin)
                                }
                            }
                        }
                        .disabled(notificationGranted)
                    }

                    Text("HealthKit (V2): sleep, HRV, and activity signals can be layered in later.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Button("Enter the Void") {
                    model.completeOnboarding()
                    Task {
                        await model.saveProfile()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.sessionToken == nil)
            }
            .padding(24)
        }
        .background(Color.black.ignoresSafeArea())
        .foregroundStyle(.white)
        .onAppear {
            startupBackend = StartupBackend.from(urlString: model.apiBaseURL)
            micGranted = AVAudioSession.sharedInstance().recordPermission == .granted
            Task { @MainActor in
                let status = await NotificationScheduler.authorizationStatus()
                notificationGranted = status == .authorized || status == .provisional || status == .ephemeral
            }
        }
    }
}

// MARK: - Void experience

struct VoidExperienceView: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var recorder = RecorderEngine()

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [Color.black, Color(red: 0.05, green: 0.06, blue: 0.1)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 24) {
                        Text(model.submissionState.title)
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.8))

                        Text(formatDuration(recorder.elapsed))
                            .font(.system(size: 44, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)

                        AudioReactiveBars(amplitude: recorder.amplitude)
                            .frame(height: 180)
                            .padding(.horizontal)

                        Button(recorder.isRecording ? "Stop & Submit" : "Start 5:00 Reflection") {
                            if recorder.isRecording {
                                stopAndSubmit()
                            } else {
                                startRecording()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(recorder.isRecording ? .red : .teal)

                        if model.submissionState == .transcribing {
                            Text("Processing transcription and insights…")
                                .foregroundStyle(.white.opacity(0.75))
                        }

                        if !model.pendingDrafts.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Pending Drafts")
                                    .font(.headline)
                                    .foregroundStyle(.white)

                                ForEach(model.pendingDrafts, id: \.path) { draft in
                                    HStack {
                                        Text(draft.lastPathComponent)
                                            .font(.footnote)
                                            .lineLimit(1)
                                        Spacer()
                                        Button("Retry") {
                                            Task {
                                                let duration = model.audioDurationForDraft(draft)
                                                await model.submitDraft(url: draft, durationSeconds: duration)
                                            }
                                        }
                                        .buttonStyle(.bordered)

                                        Button("Delete", role: .destructive) {
                                            model.deleteDraft(draft)
                                        }
                                        .buttonStyle(.bordered)
                                    }
                                    .padding(10)
                                    .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                        }

                        if let latestEntry = model.latestTranscriptEntry,
                           let transcript = latestEntry.transcript?.text {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Latest Transcript")
                                    .font(.headline)
                                    .foregroundStyle(.white)
                                Text(latestEntry.localDate)
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.7))
                                if let tags = latestEntry.insight?.moodTags, !tags.isEmpty {
                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 8) {
                                            ForEach(tags, id: \.self) { tag in
                                                TagChip(tag: tag)
                                            }
                                        }
                                    }
                                }
                                Text(transcript)
                                    .font(.footnote)
                                    .foregroundStyle(.white.opacity(0.92))
                                    .lineLimit(7)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .background(Color.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal)
                        }
                    }
                    .padding(.vertical, 24)
                }
            }
            .navigationTitle("The Void")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                recorder.onWarning = { _ in
                    UINotificationFeedbackGenerator().notificationOccurred(.warning)
                }
                recorder.onAutoStop = {
                    stopAndSubmit()
                }
                model.reloadDrafts()
            }
        }
    }

    private func startRecording() {
        Task { @MainActor in
            if recorder.recordPermissionStatus() == .denied {
                model.errorMessage = "Microphone permission is denied. Enable it in iOS Settings > Privacy & Security > Microphone."
                return
            }

            let granted = await recorder.requestPermission()
            guard granted else {
                model.errorMessage = "Microphone access is required"
                return
            }
            do {
                let draftURL = model.makeDraftURL()
                try recorder.startRecording(at: draftURL)
                model.submissionState = .recording
            } catch {
                model.errorMessage = "Failed to start recording: \(error.localizedDescription)\nIf using a Simulator, ensure an audio input is available."
            }
        }
    }

    private func stopAndSubmit() {
        guard let recordedURL = recorder.stopRecording() else {
            return
        }
        let duration = max(1, Int(recorder.elapsed))
        Task {
            await model.submitDraft(url: recordedURL, durationSeconds: duration)
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }
}

struct AudioReactiveBars: View {
    let amplitude: CGFloat

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let count = 36
            let spacing: CGFloat = 4
            let barWidth = (width - (CGFloat(count - 1) * spacing)) / CGFloat(count)

            HStack(spacing: spacing) {
                ForEach(0..<count, id: \.self) { index in
                    let phase = CGFloat(index) / CGFloat(count)
                    let wave = sin((phase + amplitude) * .pi * 4)
                    let strength = max(0.12, amplitude + wave * 0.28)
                    Capsule()
                        .fill(Color.teal.opacity(0.85))
                        .frame(width: barWidth, height: max(18, strength * 170))
                }
            }
            .frame(maxHeight: .infinity, alignment: .center)
        }
    }
}

// MARK: - Journal

struct JournalView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Mood Map")
                        .font(.headline)
                    MoodHeatmap(entries: model.entries)

                    Text("Timeline")
                        .font(.headline)

                    LazyVStack(spacing: 12) {
                        ForEach(model.entries) { entry in
                            NavigationLink(value: entry) {
                                EntryCard(entry: entry)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Journal")
            .navigationDestination(for: APIEntry.self) { entry in
                EntryDetailView(entry: entry)
            }
            .refreshable {
                await model.refreshEntries()
            }
        }
    }
}

struct MoodHeatmap: View {
    let entries: [APIEntry]

    private let columns = Array(repeating: GridItem(.fixed(14), spacing: 5), count: 14)

    var body: some View {
        // Multiple entries can exist on the same day; keep the first (latest) entry for heatmap color.
        let lookup = entries.reduce(into: [String: APIEntry]()) { result, entry in
            if result[entry.localDate] == nil {
                result[entry.localDate] = entry
            }
        }
        let dates = (0..<98).compactMap {
            Calendar.current.date(byAdding: .day, value: -$0, to: Date())
        }

        LazyVGrid(columns: columns, spacing: 5) {
            ForEach(dates, id: \.self) { day in
                let key = DateFormatter.localDate.string(from: day)
                let moodScore = lookup[key]?.insight?.moodScore
                RoundedRectangle(cornerRadius: 2)
                    .fill(color(for: moodScore))
                    .frame(width: 14, height: 14)
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
    }

    private func color(for moodScore: Double?) -> Color {
        guard let moodScore else {
            return Color.gray.opacity(0.18)
        }
        switch moodScore {
        case ..<(-1.0): return Color(red: 0.4, green: 0.12, blue: 0.2)
        case ..<(-0.2): return Color(red: 0.6, green: 0.28, blue: 0.3)
        case ..<0.3: return Color(red: 0.35, green: 0.36, blue: 0.38)
        case ..<1.1: return Color(red: 0.28, green: 0.55, blue: 0.49)
        default: return Color(red: 0.13, green: 0.65, blue: 0.58)
        }
    }
}

struct EntryCard: View {
    let entry: APIEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(entry.localDate)
                    .font(.headline)
                Spacer()
                Text(entry.status.capitalized)
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.15), in: Capsule())
            }

            if let tags = entry.insight?.moodTags, !tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(tags, id: \.self) { tag in
                            TagChip(tag: tag)
                        }
                    }
                }
            }

            if let summary = entry.insight?.summary {
                Text(summary)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            if let transcript = entry.transcript?.text, !transcript.isEmpty {
                Text(transcript)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
        }
        .padding(14)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

struct EntryDetailView: View {
    @EnvironmentObject private var model: AppModel
    let entry: APIEntry

    @StateObject private var audioPlayback = AudioPlaybackController()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text(entry.localDate)
                    .font(.title2.bold())

                if let insight = entry.insight {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Summary")
                            .font(.headline)
                        Text(insight.summary)

                        Text("Mood Score: \(String(format: "%.1f", insight.moodScore))")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Text("Tags")
                            .font(.headline)
                        WrapTags(tags: insight.moodTags)
                    }
                }

                if let transcript = entry.transcript {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Transcript")
                            .font(.headline)
                        Text(transcript.text)
                            .font(.body)
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Audio")
                        .font(.headline)

                    HStack(spacing: 10) {
                        Button(audioPlayback.isReady ? "Reload" : "Load Audio") {
                            Task {
                                await loadAudio(forceReload: true)
                            }
                        }
                        .buttonStyle(.bordered)

                        Button(audioPlayback.isPlaying ? "Pause" : "Play") {
                            audioPlayback.togglePlayback()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!audioPlayback.isReady || audioPlayback.isLoading)

                        Button("Restart") {
                            audioPlayback.restart()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!audioPlayback.isReady || audioPlayback.isLoading)
                    }

                    if audioPlayback.isLoading {
                        ProgressView("Loading audio...")
                    }

                    Slider(
                        value: Binding(
                            get: { audioPlayback.currentTime },
                            set: { audioPlayback.scrub(to: $0) }
                        ),
                        in: 0...max(audioPlayback.duration, 1),
                        onEditingChanged: { editing in
                            if editing {
                                audioPlayback.beginScrubbing()
                            } else {
                                audioPlayback.endScrubbing()
                            }
                        }
                    )
                    .disabled(!audioPlayback.isReady)

                    HStack {
                        Text(formatDuration(audioPlayback.currentTime))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(formatDuration(audioPlayback.duration))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Entry")
        .task {
            await loadAudio(forceReload: false)
        }
        .onDisappear {
            audioPlayback.stop()
        }
    }

    private func loadAudio(forceReload: Bool) async {
        do {
            try await audioPlayback.load(
                fetchAudio: { try await model.fetchAudio(entryID: entry.id) },
                forceReload: forceReload
            )
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded()))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }
}

struct WrapTags: View {
    let tags: [String]

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 8)], spacing: 8) {
            ForEach(tags, id: \.self) { tag in
                TagChip(tag: tag)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

struct TagChip: View {
    let tag: String

    private var label: String {
        tag.split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    var body: some View {
        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                LinearGradient(
                    colors: [
                        Color.teal.opacity(0.28),
                        Color.cyan.opacity(0.16),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: Capsule()
            )
            .overlay(
                Capsule()
                    .stroke(Color.teal.opacity(0.35), lineWidth: 1)
            )
    }
}

// MARK: - Social

struct SocialView: View {
    @EnvironmentObject private var model: AppModel
    @State private var revealNameOverride: String = ""
    @State private var acceptInviteToken: String = ""

    private let columns = [GridItem(.adaptive(minimum: 74), spacing: 16)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Friend Dots")
                        .font(.headline)
                    Text("You are @\(model.anonymousHandle)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if model.socialDots.isEmpty {
                        Text("No friends linked yet. Create an invite below and accept it from a second account/simulator.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    LazyVGrid(columns: columns, spacing: 16) {
                        ForEach(model.socialDots) { dot in
                            VStack(spacing: 8) {
                                Circle()
                                    .fill(Color(hex: dot.dotColor))
                                    .frame(width: 40, height: 40)
                                Text(dot.label ?? "Anonymous")
                                    .font(.caption)
                                    .lineLimit(1)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                            .onTapGesture {
                                if model.revealFriendIDs.contains(dot.userId) {
                                    model.revealFriendIDs.remove(dot.userId)
                                } else {
                                    model.revealFriendIDs.insert(dot.userId)
                                }
                            }
                            .overlay(alignment: .topTrailing) {
                                if model.revealFriendIDs.contains(dot.userId) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.teal)
                                        .padding(6)
                                }
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Reveal Mode")
                            .font(.headline)

                        Picker("Reveal", selection: $model.revealMode) {
                            ForEach(RevealModeOption.allCases) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)

                        if model.revealMode == .revealedToSpecific {
                            Text("Tap dots above to select which friends see your label today.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }

                        TextField("Display name override", text: $revealNameOverride)
                            .textFieldStyle(.roundedBorder)

                        Button("Save Today’s Reveal") {
                            Task {
                                await model.saveRevealMode(displayNameOverride: revealNameOverride)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Social Testing")
                            .font(.headline)
                        Text("Current backend: \(model.apiBaseURL)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Text("1. On Device A, sign in with dev token: dev-user-a")
                            .font(.footnote)
                        Text("2. On Device B/Simulator B, sign in with: dev-user-b")
                            .font(.footnote)
                        Text("3. Create invite on A, paste invite link/token on B, then submit entries on both.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Friend Invites")
                            .font(.headline)

                        Button("Create Invite Link") {
                            Task { await model.createInvite() }
                        }
                        .buttonStyle(.bordered)

                        if let inviteURL = model.inviteURL {
                            HStack {
                                Text(inviteURL)
                                    .font(.footnote)
                                    .textSelection(.enabled)
                                Spacer()
                                Button("Copy") {
                                    UIPasteboard.general.string = inviteURL
                                }
                                .buttonStyle(.bordered)
                            }
                        }

                        if let inviteToken = model.inviteToken {
                            HStack {
                                Text(inviteToken)
                                    .font(.footnote.monospaced())
                                    .textSelection(.enabled)
                                Spacer()
                                Button("Copy Token") {
                                    UIPasteboard.general.string = inviteToken
                                }
                                .buttonStyle(.bordered)
                            }
                        }

                        TextField("Paste invite link or token", text: $acceptInviteToken)
                            .textFieldStyle(.roundedBorder)
                        Button("Accept Invite") {
                            Task {
                                await model.acceptInvite(token: acceptInviteToken)
                                acceptInviteToken = ""
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding()
            }
            .navigationTitle("Social")
            .refreshable {
                await model.refreshSocialDots()
            }
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Display name", text: $model.displayName)
                    Text("Anonymous handle: @\(model.anonymousHandle)")
                }

                Section("Check-In") {
                    DatePicker("Daily time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
                    Toggle("Notifications enabled", isOn: $model.notificationEnabled)
                    Button("Schedule daily reminder") {
                        Task {
                            do {
                                _ = await NotificationScheduler.requestPermission()
                                try await NotificationScheduler.scheduleDaily(at: model.dailyCheckin)
                            } catch {
                                model.errorMessage = error.localizedDescription
                            }
                        }
                    }
                }

                Section("Integrations") {
                    Text("HealthKit (V2 planned): sleep, HRV, and activity.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Networking") {
                    TextField("API base URL", text: $model.apiBaseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Button("Save API URL") {
                        model.applyAPIBaseURL()
                    }
                    Button("Test API Connection") {
                        Task {
                            await model.testAPIConnection()
                        }
                    }
                    if let status = model.apiConnectionStatus {
                        Text(status)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Text("Simulator: http://127.0.0.1:8080")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Text("Device: http://<your-mac-lan-ip>:8080 and run API with --host 0.0.0.0")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Actions") {
                    Button("Save Settings") {
                        Task {
                            await model.saveProfile()
                        }
                    }
                    Button("Refresh") {
                        Task {
                            await model.refreshAll()
                        }
                    }
                    Button("Sign Out", role: .destructive) {
                        model.signOut()
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

// MARK: - Helpers

extension Color {
    init(hex: String) {
        var sanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        sanitized = sanitized.replacingOccurrences(of: "#", with: "")

        var value: UInt64 = 0
        Scanner(string: sanitized).scanHexInt64(&value)

        let r, g, b: Double
        if sanitized.count == 6 {
            r = Double((value & 0xFF0000) >> 16) / 255.0
            g = Double((value & 0x00FF00) >> 8) / 255.0
            b = Double(value & 0x0000FF) / 255.0
        } else {
            r = 0.5
            g = 0.5
            b = 0.5
        }

        self.init(red: r, green: g, blue: b)
    }
}

extension DateFormatter {
    static let hhmm: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        return formatter
    }()

    static let localDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()
}

#Preview {
    ContentView()
}
