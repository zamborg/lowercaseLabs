//
//  ContentView.swift
//  theVoid
//
//  Created by zubin aysola on 2/16/26.
//

import AuthenticationServices
import AVFoundation
import CryptoKit
import Speech
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
    let title: String
    let transcript: APITranscript?
    let insight: APIInsight?

    var displayTitle: String {
        Self.sanitizeTitle(title)
    }

    var createdAtTimeLabel: String? {
        guard let date = parsedCreatedAt else { return nil }
        return DateFormatter.clock.string(from: date)
    }

    init(
        id: String,
        localDate: String,
        durationSeconds: Int,
        status: String,
        createdAt: String,
        transcript: APITranscript?,
        insight: APIInsight?,
        title: String = "Entry"
    ) {
        self.id = id
        self.localDate = localDate
        self.durationSeconds = durationSeconds
        self.status = status
        self.createdAt = createdAt
        self.title = Self.sanitizeTitle(title)
        self.transcript = transcript
        self.insight = insight
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case localDate
        case durationSeconds
        case status
        case createdAt
        case title
        case transcript
        case insight
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        localDate = try container.decode(String.self, forKey: .localDate)
        durationSeconds = try container.decode(Int.self, forKey: .durationSeconds)
        status = try container.decode(String.self, forKey: .status)
        createdAt = try container.decode(String.self, forKey: .createdAt)
        title = Self.sanitizeTitle(try container.decodeIfPresent(String.self, forKey: .title) ?? "Entry")
        transcript = try container.decodeIfPresent(APITranscript.self, forKey: .transcript)
        insight = try container.decodeIfPresent(APIInsight.self, forKey: .insight)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(localDate, forKey: .localDate)
        try container.encode(durationSeconds, forKey: .durationSeconds)
        try container.encode(status, forKey: .status)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(Self.sanitizeTitle(title), forKey: .title)
        try container.encodeIfPresent(transcript, forKey: .transcript)
        try container.encodeIfPresent(insight, forKey: .insight)
    }

    static func sanitizeTitle(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Entry" }
        let collapsed = trimmed.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )
        if collapsed.count <= 48 {
            return collapsed
        }
        return String(collapsed.prefix(48))
    }

    private var parsedCreatedAt: Date? {
        if let date = Self.iso8601WithFractional.date(from: createdAt) {
            return date
        }
        return Self.iso8601Basic.date(from: createdAt)
    }

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let iso8601Basic: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}

struct APISocialDot: Codable, Identifiable, Hashable {
    var id: String { userId }
    let userId: String
    let dotColor: String
    let dotTags: [String]?
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

enum ReminderWeekday: Int, CaseIterable, Identifiable {
    case monday = 2
    case tuesday = 3
    case wednesday = 4
    case thursday = 5
    case friday = 6
    case saturday = 7
    case sunday = 1

    var id: Int { rawValue }

    var shortTitle: String {
        switch self {
        case .monday: return "Mon"
        case .tuesday: return "Tue"
        case .wednesday: return "Wed"
        case .thursday: return "Thu"
        case .friday: return "Fri"
        case .saturday: return "Sat"
        case .sunday: return "Sun"
        }
    }

    var title: String {
        switch self {
        case .monday: return "Monday"
        case .tuesday: return "Tuesday"
        case .wednesday: return "Wednesday"
        case .thursday: return "Thursday"
        case .friday: return "Friday"
        case .saturday: return "Saturday"
        case .sunday: return "Sunday"
        }
    }

    static var ordered: [ReminderWeekday] {
        [.monday, .tuesday, .wednesday, .thursday, .friday, .saturday, .sunday]
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
    static let productionBaseURLString = "https://thevoid-local.fly.dev"
    static let defaultBaseURLString = BackendClient.productionBaseURLString

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

    struct UpdateSocialDotPayload: Encodable {
        let moodScore: Double?
        let moodTags: [String]
        let dotColor: String?
    }

    struct InvitePayload: Encodable {
        let expiresInDays: Int
        let maxUses: Int
    }

    struct AcceptInvitePayload: Encodable {
        let token: String
    }

    struct FeedbackPayload: Encodable {
        let kind: String
        let message: String
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

    func fetchSocialDots(token: String, localDate: String? = nil) async throws -> APISocialDotsEnvelope {
        let path: String
        if let localDate, !localDate.isEmpty {
            path = "/social/dots?local_date=\(localDate)"
        } else {
            path = "/social/dots"
        }
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "GET", token: token)
        return try await send(request, decode: APISocialDotsEnvelope.self)
    }

    func publishLocalDot(
        token: String,
        localDate: String,
        moodScore: Double,
        moodTags: [String],
        dotColor: String?
    ) async throws {
        let payload = UpdateSocialDotPayload(
            moodScore: max(-2, min(moodScore, 2)),
            moodTags: Array(moodTags.prefix(8)),
            dotColor: dotColor
        )
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/social/presence/\(localDate)/dot", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "PUT", token: token, body: body)
        _ = try await send(request, decode: APIMessage.self)
    }

    func deleteLocalDot(token: String, localDate: String) async throws {
        guard let url = URL(string: "/social/presence/\(localDate)/dot", relativeTo: baseURL) else {
            throw APIError.invalidURL
        }
        let request = buildRequest(url: url, method: "DELETE", token: token)
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

    func submitFeedback(token: String, kind: String, message: String) async throws {
        let payload = FeedbackPayload(kind: kind, message: message)
        let body = try encoder.encode(payload)
        guard let url = URL(string: "/feedback", relativeTo: baseURL) else {
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
    var onAutoStop: ((URL, Int) -> Void)?

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
            let duration = max(1, Int(elapsed.rounded()))
            if let finishedURL = stopRecording() {
                onAutoStop?(finishedURL, duration)
            }
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
    struct WeeklyReminder: Hashable {
        let weekday: ReminderWeekday
        let time: Date
    }

    private static let legacyIdentifier = "void.daily.checkin"
    private static let weeklyIdentifierPrefix = "void.weekly.checkin."

    private static var weeklyIdentifiers: [String] {
        ReminderWeekday.ordered.map { "\(weeklyIdentifierPrefix)\($0.rawValue)" }
    }

    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
                continuation.resume(returning: granted)
            }
        }
    }

    static func scheduleDaily(at date: Date) async throws {
        try await scheduleWeekly(
            reminders: ReminderWeekday.ordered.map { WeeklyReminder(weekday: $0, time: date) }
        )
    }

    static func scheduleWeekly(reminders: [WeeklyReminder]) async throws {
        let center = UNUserNotificationCenter.current()
        center.removePendingNotificationRequests(withIdentifiers: [legacyIdentifier] + weeklyIdentifiers)

        for reminder in reminders {
            let content = UNMutableNotificationContent()
            content.title = "Step Into the Void"
            content.body = "Your one intentional check-in is ready."
            content.sound = .default

            let hm = Calendar.current.dateComponents([.hour, .minute], from: reminder.time)
            var components = DateComponents()
            components.weekday = reminder.weekday.rawValue
            components.hour = hm.hour
            components.minute = hm.minute

            let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
            let identifier = "\(weeklyIdentifierPrefix)\(reminder.weekday.rawValue)"
            let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)

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

    @Published var inviteURL: String?
    @Published var inviteToken: String?
    @Published var errorMessage: String?

    private let client = BackendClient()
    private let draftStore = DraftStore()
    private let localStore = LocalJournalStore()
    private var liquidModelPreparationTask: Task<Void, Never>?
    private var liquidModelPreparationOperationID: UUID?

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
    }

    var needsOnboarding: Bool {
        sessionToken == nil || !onboardingComplete
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
        apiBaseURL = BackendClient.productionBaseURLString
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

        try? client.updateBaseURL(apiBaseURL)
        liquidModelPrepared = defaults.bool(forKey: Keys.liquidModelPrepared)
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
        onboardingComplete = false
        persistState()
    }

    func refreshAll() async {
        await refreshEntries()
        await refreshSocialDots()
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
            let envelope = try await client.fetchSocialDots(token: sessionToken)
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

    func submitDraft(url: URL, durationSeconds: Int, shareToSocial: Bool = true) async {
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
            let localDate = DateFormatter.localDate.string(from: Date())
            submissionState = .transcribing
            lastInsightRuntimeSummary = nil
            let (transcript, insight) = await LocalReflectionAnalyzer.analyze(
                audioURL: url,
                durationSeconds: normalizedDuration,
                useLiquidInsights: liquidModelPrepared
            )
            lastInsightRuntimeSummary = Self.insightRuntimeSummary(from: transcript?.providerMetadata)
            let stored = try localStore.saveReflection(
                for: userID,
                draftURL: url,
                durationSeconds: normalizedDuration,
                transcript: transcript,
                insight: insight,
                localDate: localDate,
                wasSharedToSocial: shareToSocial
            )
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
                    errorMessage = "Saved locally, but could not sync social dot: \(error.localizedDescription)"
                }
            } else if shareToSocial {
                errorMessage = "Saved audio note locally without tags. Retranscribe from the entry to generate dots."
            }

            await refreshEntries()
            await refreshSocialDots()
            reloadDrafts()
        } catch {
            submissionState = .failed
            errorMessage = error.localizedDescription
            reloadDrafts()
        }
    }

    func redownloadLiquidModel() {
        startLiquidModelPreparation(forceRedownload: true)
    }

    func retryLiquidModelPreparation() {
        startLiquidModelPreparation(forceRedownload: false)
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
                    errorMessage = "Tags updated locally, but social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = error.localizedDescription
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
            await refreshEntries()
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
            let (transcript, insight) = await LocalReflectionAnalyzer.analyze(
                audioURL: audioURL,
                durationSeconds: durationSeconds,
                useLiquidInsights: liquidModelPrepared
            )
            guard let transcript else {
                errorMessage = "Could not transcribe this audio note. Try again."
                return nil
            }

            let updateResult = try localStore.updateEntryAnalysis(
                for: userID,
                entryID: entryID,
                transcript: transcript,
                insight: insight
            )

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
                    errorMessage = "Retranscribed locally, but social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            if updateResult.wasSharedToSocial {
                await refreshSocialDots()
            }

            return entries.first(where: { $0.id == entryID }) ?? updateResult.updatedEntry
        } catch {
            errorMessage = error.localizedDescription
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
                    errorMessage = "Entry deleted locally, but social sync failed: \(error.localizedDescription)"
                }
            }

            await refreshEntries()
            await refreshSocialDots()
        } catch {
            errorMessage = error.localizedDescription
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
            errorMessage = "\(error.localizedDescription)\nAPI: \(apiBaseURL)"
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

    private static func defaultReminderTimes(base: Date) -> [Int: Date] {
        ReminderWeekday.ordered.reduce(into: [Int: Date]()) { partialResult, day in
            partialResult[day.rawValue] = base
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
            } else if model.showsLiquidModelPreparationScreen {
                LiquidModelPreparationView()
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

struct LiquidModelPreparationView: View {
    @EnvironmentObject private var model: AppModel

    private var progress: Double {
        max(0, min(1, model.liquidModelPreparationProgress))
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(red: 0.03, green: 0.08, blue: 0.12)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 22) {
                Spacer()

                Text("Preparing On-Device AI")
                    .font(.system(size: 31, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text("Downloading Liquid model to this device.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.74))

                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.16), lineWidth: 12)

                    Circle()
                        .trim(from: 0, to: max(progress, 0.01))
                        .stroke(
                            AngularGradient(
                                colors: [
                                    Color.cyan.opacity(0.95),
                                    Color.teal.opacity(0.95),
                                    Color.cyan.opacity(0.95),
                                ],
                                center: .center
                            ),
                            style: StrokeStyle(lineWidth: 12, lineCap: .round)
                        )
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.24), value: progress)

                    Text("\(Int((progress * 100).rounded()))%")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
                .frame(width: 148, height: 148)

                Text(model.liquidModelPreparationStatus)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.70))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)

                if let error = model.liquidModelPreparationError {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red.opacity(0.9))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)

                    Button("Retry Download") {
                        model.retryLiquidModelPreparation()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.teal)
                }

                VStack(spacing: 10) {
                    Text("Without the model the insights engine will not work, you can always download this later from settings.")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.72))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 340)

                    Button("Cancel") {
                        model.cancelLiquidModelPreparation()
                    }
                    .buttonStyle(.bordered)
                    .tint(.white.opacity(0.88))
                }
                .padding(.top, 6)

                Spacer()
            }
            .padding(24)
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
            
            EmotionAtlasScreen()
                .tabItem {
                    Label("Atlas", systemImage: "sparkles.rectangle.stack")
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

    @State private var micGranted = false
    @State private var speechGranted = false
    @State private var currentAppleNonce: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("theVoid")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                Text("One intentional check-in. Private by default.")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 12) {
                    Text("Identity")
                        .font(.headline)

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

                    if !model.anonymousHandle.isEmpty {
                        Text("Signed in as @\(model.anonymousHandle)")
                            .font(.subheadline)
                    }
                    
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

//                VStack(alignment: .leading, spacing: 12) {
//                    Text("Check-In")
//                        .font(.headline)
//                    DatePicker("Default time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
//                    ReminderScheduleEditor()
//
//                    Button("Save Reminder Schedule") {
//                        Task { await model.configureDailyReminder() }
//                    }
//                    .buttonStyle(.borderedProminent)
//
//                    if let reminderStatus = model.reminderStatus {
//                        Text(reminderStatus)
//                            .font(.footnote)
//                            .foregroundStyle(.secondary)
//                    }
//
//                }

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
                        Label("Speech Recognition", systemImage: speechGranted ? "checkmark.circle.fill" : "circle")
                        Spacer()
                        Button(speechGranted ? "Granted" : "Allow") {
                            Task {
                                speechGranted = await LocalReflectionAnalyzer.requestSpeechPermission()
                            }
                        }
                        .disabled(speechGranted)
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
            if model.apiBaseURL != BackendClient.productionBaseURLString {
                model.apiBaseURL = BackendClient.productionBaseURLString
                model.applyAPIBaseURL()
            }
            micGranted = AVAudioSession.sharedInstance().recordPermission == .granted
            speechGranted = SFSpeechRecognizer.authorizationStatus() == .authorized
        }
    }
}

// MARK: - Void experience

struct VoidExperienceView: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var recorder = RecorderEngine()
    @State private var pendingSubmission: PendingSubmission?
    @State private var showDecisionSheet = false
    @State private var showWelcomeOverlay = false
    @State private var autoWelcomePendingAck = false
    @State private var isRecordingLocked = false

    private struct PendingSubmission {
        let url: URL
        let durationSeconds: Int
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                let availableHeight = geometry.size.height
                let isCompactHeight = availableHeight < 760
                let isVeryCompactHeight = availableHeight < 700
                let stackSpacing: CGFloat = isVeryCompactHeight ? 14 : (isCompactHeight ? 18 : 26)
                let signalSize: CGFloat = isVeryCompactHeight ? 156 : (isCompactHeight ? 172 : 190)
                let touchAreaMinHeight: CGFloat = isVeryCompactHeight ? 180 : (isCompactHeight ? 200 : 240)
                let topPadding: CGFloat = isVeryCompactHeight ? 16 : (isCompactHeight ? 20 : 26)
                let bottomPadding = max(6, geometry.safeAreaInsets.bottom + 4)

                ZStack {
                    LinearGradient(
                        colors: [Color.black, Color(red: 0.05, green: 0.06, blue: 0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .ignoresSafeArea()

                    VStack(spacing: stackSpacing) {
                        Text(model.submissionState.title)
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.78))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .frame(maxWidth: .infinity)
                            .multilineTextAlignment(.center)

                        if !model.liquidModelPrepared {
                            VStack(spacing: 6) {
                                Label("Insights model not downloaded", systemImage: "exclamationmark.triangle.fill")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(Color.orange.opacity(0.95))
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.85)
                                Text("Without the model the insights engine will not work. Download it from Settings > On-Device AI to create dots.")
                                    .font(.footnote)
                                    .foregroundStyle(.white.opacity(0.74))
                                    .multilineTextAlignment(.center)
                                    .lineLimit(isVeryCompactHeight ? 3 : 4)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .layoutPriority(2)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.horizontal, 12)
                        }

                        Text(formatDuration(recorder.elapsed))
                            .font(.system(size: isVeryCompactHeight ? 40 : 46, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)

                        PulsingSignalDot(
                            amplitude: recorder.amplitude,
                            isRecording: recorder.isRecording,
                            isProcessing: model.submissionState == .transcribing
                        )
                        .frame(width: signalSize, height: signalSize)

                        Text(recordingInstructionTitle)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.9))
                            .multilineTextAlignment(.center)

                        Text(recordingInstructionDetail)
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.68))
                            .lineLimit(2)
                            .multilineTextAlignment(.center)

                        if model.submissionState == .transcribing {
                            Text("Processing transcription and tags on this device…")
                                .font(.footnote)
                                .foregroundStyle(.white.opacity(0.75))
                                .multilineTextAlignment(.center)
                                .lineLimit(2)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        GeometryReader { padGeometry in
                            let dynamicTouchAreaHeight = max(touchAreaMinHeight, padGeometry.size.height)
                            touchAndHoldPad
                                .frame(height: dynamicTouchAreaHeight)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                        }
                        .frame(maxHeight: .infinity, alignment: .bottom)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, topPadding)
                    .padding(.bottom, bottomPadding)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .blur(radius: showWelcomeOverlay ? 6 : 0)
                    .allowsHitTesting(!showWelcomeOverlay)

                    if showWelcomeOverlay {
                        LinearGradient(
                            colors: [
                                Color.black.opacity(0.72),
                                Color.black.opacity(0.58),
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                            .ignoresSafeArea()

                        WelcomeOverlayCard {
                            dismissWelcomeOverlay()
                        }
                        .padding(.horizontal, 22)
                        .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }
                }
            }
            .navigationTitle("The Void")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        autoWelcomePendingAck = false
                        showWelcomeOverlay = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                    }
                }
            }
            .sheet(isPresented: $showDecisionSheet, onDismiss: { pendingSubmission = nil }) {
                RecordingDecisionSheet(
                    onShare: { submitPending(shareToSocial: true) },
                    onLocalOnly: { submitPending(shareToSocial: false) }
                )
                .presentationDetents([.fraction(0.32)])
                .presentationDragIndicator(.visible)
            }
            .onAppear {
                recorder.onWarning = { _ in
                    UINotificationFeedbackGenerator().notificationOccurred(.warning)
                }
                recorder.onAutoStop = { url, duration in
                    pendingSubmission = PendingSubmission(url: url, durationSeconds: duration)
                    model.submissionState = .uploading
                    showDecisionSheet = true
                    isRecordingLocked = false
                }
                model.reloadDrafts()
                maybePresentWelcomeOverlayIfNeeded()
            }
            .animation(.easeInOut(duration: 0.2), value: showWelcomeOverlay)
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

    private func finalizeRecordingForChoice() {
        guard let recordedURL = recorder.stopRecording() else {
            return
        }
        isRecordingLocked = false
        let duration = max(1, Int(recorder.elapsed))
        pendingSubmission = PendingSubmission(url: recordedURL, durationSeconds: duration)
        model.submissionState = .uploading
        showDecisionSheet = true
    }

    private func submitPending(shareToSocial: Bool) {
        guard let pendingSubmission else {
            return
        }
        let payload = pendingSubmission
        self.pendingSubmission = nil
        showDecisionSheet = false
        Task {
            await model.submitDraft(
                url: payload.url,
                durationSeconds: payload.durationSeconds,
                shareToSocial: shareToSocial
            )
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }

    private var welcomeSeenKey: String {
        let scopedID = model.userID.isEmpty ? "anonymous" : model.userID
        return "thevoid.welcome.overlay.seen.\(scopedID)"
    }

    private func maybePresentWelcomeOverlayIfNeeded() {
        guard !model.userID.isEmpty else {
            return
        }
        if UserDefaults.standard.bool(forKey: welcomeSeenKey) {
            return
        }
        autoWelcomePendingAck = true
        showWelcomeOverlay = true
    }

    private func dismissWelcomeOverlay() {
        if autoWelcomePendingAck {
            UserDefaults.standard.set(true, forKey: welcomeSeenKey)
            autoWelcomePendingAck = false
        }
        showWelcomeOverlay = false
    }

    private var recordingInstructionTitle: String {
        if recorder.isRecording {
            return isRecordingLocked
                ? "Recording locked. Touch the pad to stop"
                : "Release on the pad to stop. Release outside to lock"
        }
        return "Press and hold to record"
    }

    private var recordingInstructionDetail: String {
        if recorder.isRecording {
            return isRecordingLocked
                ? "Recording continues until you touch the pad again."
                : "Slide outside and lift to keep recording hands-free."
        }
        return "Max 5:00."
    }

    private var touchAndHoldPad: some View {
        RoundedRectangle(cornerRadius: 22)
            .fill(
                LinearGradient(
                    colors: [
                        Color.white.opacity(recorder.isRecording ? 0.22 : 0.11),
                        Color.white.opacity(recorder.isRecording ? 0.14 : 0.06),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .frame(maxWidth: .infinity)
            .overlay {
                VStack(spacing: 10) {
                    Image(systemName: recorder.isRecording ? (isRecordingLocked ? "lock.fill" : "waveform.circle.fill") : "hand.tap.fill")
                        .font(.system(size: 38, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.88))
                    Text(
                        recorder.isRecording
                            ? (isRecordingLocked ? "Recording Locked" : "Recording…")
                            : "Touch and hold here"
                    )
                        .font(.headline)
                        .foregroundStyle(.white.opacity(0.92))
                        .lineLimit(1)
                        .minimumScaleFactor(0.85)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 22)
                    .stroke(Color.white.opacity(recorder.isRecording ? 0.5 : 0.2), lineWidth: 1.2)
            )
            .overlay {
                PressAndHoldCaptureView(
                    onPressStart: {
                        guard !showDecisionSheet else { return }
                        if recorder.isRecording {
                            if isRecordingLocked {
                                finalizeRecordingForChoice()
                            }
                            return
                        }
                        isRecordingLocked = false
                        startRecording()
                    },
                    onPressEnd: { endedInside in
                        guard recorder.isRecording else { return }
                        guard !isRecordingLocked else { return }
                        if endedInside {
                            finalizeRecordingForChoice()
                        } else {
                            isRecordingLocked = true
                        }
                    }
                )
            }
            .contentShape(RoundedRectangle(cornerRadius: 22))
    }
}

private struct PressAndHoldCaptureView: UIViewRepresentable {
    let onPressStart: () -> Void
    let onPressEnd: (Bool) -> Void

    func makeUIView(context: Context) -> PressAndHoldControl {
        let control = PressAndHoldControl()
        control.onPressStart = onPressStart
        control.onPressEnd = onPressEnd
        return control
    }

    func updateUIView(_ uiView: PressAndHoldControl, context _: Context) {
        uiView.onPressStart = onPressStart
        uiView.onPressEnd = onPressEnd
    }
}

private final class PressAndHoldControl: UIControl {
    var onPressStart: (() -> Void)?
    var onPressEnd: ((Bool) -> Void)?
    private var isPressing = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        isExclusiveTouch = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func beginTracking(_ touch: UITouch, with event: UIEvent?) -> Bool {
        _ = touch
        _ = event
        guard !isPressing else { return true }
        isPressing = true
        onPressStart?()
        return true
    }

    override func continueTracking(_ touch: UITouch, with event: UIEvent?) -> Bool {
        _ = touch
        _ = event
        return true
    }

    override func endTracking(_ touch: UITouch?, with event: UIEvent?) {
        _ = event
        let endedInside: Bool
        if let touch {
            endedInside = bounds.contains(touch.location(in: self))
        } else {
            endedInside = false
        }
        finishPressIfNeeded(endedInside: endedInside)
    }

    override func cancelTracking(with event: UIEvent?) {
        _ = event
        finishPressIfNeeded(endedInside: false)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesCancelled(touches, with: event)
        finishPressIfNeeded(endedInside: false)
    }

    private func finishPressIfNeeded(endedInside: Bool) {
        guard isPressing else { return }
        isPressing = false
        onPressEnd?(endedInside)
    }
}

struct PulsingSignalDot: View {
    let amplitude: CGFloat
    let isRecording: Bool
    let isProcessing: Bool

    var body: some View {
        let normalized = max(0.08, min(1.0, amplitude))
        let pulseDriver: CGFloat = isRecording ? normalized : (isProcessing ? 0.58 : 0.16)
        let pulseScale = 0.86 + (pulseDriver * 0.54)
        let haloColor = isProcessing ? Color.orange : Color.teal
        let coreTop = isProcessing ? Color(red: 1.0, green: 0.78, blue: 0.32) : Color.cyan
        let coreBottom = isProcessing ? Color(red: 0.98, green: 0.56, blue: 0.18) : Color.teal
        let isActive = isRecording || isProcessing

        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            haloColor.opacity(isActive ? 0.24 : 0.10),
                            Color.clear,
                        ],
                        center: .center,
                        startRadius: 12,
                        endRadius: 102
                    )
                )
                .scaleEffect(isActive ? pulseScale * 1.36 : 1.0)

            Circle()
                .fill(
                    LinearGradient(
                        colors: [
                            coreTop.opacity(isActive ? 0.92 : 0.72),
                            coreBottom.opacity(isActive ? 0.74 : 0.52),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 92, height: 92)
                .scaleEffect(isActive ? pulseScale : 0.88)
                .shadow(color: haloColor.opacity(0.45), radius: isActive ? 14 : 7, x: 0, y: 1)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.28), lineWidth: 1)
                )
        }
        .animation(.easeInOut(duration: 0.12), value: normalized)
        .animation(.easeInOut(duration: 0.18), value: isRecording)
        .animation(.easeInOut(duration: 0.22), value: isProcessing)
    }
}

struct RecordingDecisionSheet: View {
    let onShare: () -> Void
    let onLocalOnly: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Text("Save This Reflection")
                .font(.headline)
            HStack(spacing: 14) {
                decisionButton(
                    icon: "person.2.circle.fill",
                    title: "Share Dot",
                    subtitle: "Post to friends",
                    tint: .teal,
                    action: onShare
                )
                decisionButton(
                    icon: "iphone.gen3.radiowaves.left.and.right",
                    title: "Local Only",
                    subtitle: "Keep it private",
                    tint: .indigo,
                    action: onLocalOnly
                )
            }
            Text("Shared entries publish the color wheel dot to all friends.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(20)
    }

    private func decisionButton(
        icon: String,
        title: String,
        subtitle: String,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

struct WelcomeOverlayCard: View {
    let onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("WELCOME")
                        .font(.caption.weight(.semibold))
                        .tracking(1.4)
                        .foregroundStyle(Color.white.opacity(0.58))
                    Text("theVoid")
                        .font(.system(size: 40, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                }
                Spacer()
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white.opacity(0.82))
                        .frame(width: 30, height: 30)
                        .background(Color.white.opacity(0.08), in: Circle())
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.18), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }

            Text("theVoid is a place for you to share your thoughts freely. Your thoughts become dots, a reflection of your mood in that moment.")
                .font(.body.weight(.medium))
                .foregroundStyle(Color.white.opacity(0.93))

            Text("Everything is transcribed and analyzed on your device. You choose if you want to share your *dots* with friends.")
                .font(.body)
                .foregroundStyle(Color.white.opacity(0.86))

            HStack(spacing: 8) {
                WelcomeBadge(icon: "lock.shield.fill", label: "On-device")
                WelcomeBadge(icon: "circle.grid.2x2.fill", label: "Dots, not transcripts")
                WelcomeBadge(icon: "sparkles", label: "Check in anytime")
            }

            Text("Check in as much as you'd like, and have a fun time in theVoid.")
                .font(.callout)
                .foregroundStyle(Color.white.opacity(0.86))

            Button {
                onDismiss()
            } label: {
                HStack {
                    Spacer()
                    Text("Enter theVoid")
                        .font(.headline.weight(.semibold))
                    Spacer()
                }
                .padding(.vertical, 12)
                .background(
                    LinearGradient(
                        colors: [
                            Color.cyan.opacity(0.92),
                            Color.teal.opacity(0.82),
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: RoundedRectangle(cornerRadius: 12)
                )
                .foregroundStyle(.black.opacity(0.78))
            }
            .buttonStyle(.plain)
        }
        .padding(22)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.05, green: 0.08, blue: 0.14),
                                Color(red: 0.08, green: 0.12, blue: 0.19),
                                Color(red: 0.04, green: 0.09, blue: 0.13),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Circle()
                    .fill(Color.teal.opacity(0.26))
                    .frame(width: 180, height: 180)
                    .offset(x: 78, y: -76)
                    .blur(radius: 32)

                Circle()
                    .fill(Color.cyan.opacity(0.18))
                    .frame(width: 130, height: 130)
                    .offset(x: -84, y: 82)
                    .blur(radius: 26)
            }
        )
        .overlay(
            RoundedRectangle(cornerRadius: 24)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.5), radius: 24, x: 0, y: 14)
    }
}

private struct WelcomeBadge: View {
    let icon: String
    let label: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption.weight(.bold))
            Text(label)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .foregroundStyle(Color.white.opacity(0.86))
        .background(Color.white.opacity(0.10), in: Capsule())
        .overlay(
            Capsule()
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
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

struct EmotionAtlasScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    EmotionAtlasView(entries: model.entries, height: 500)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Tags")
                            .font(.headline)
                        let recent = recentTags(from: model.entries)
                        if recent.isEmpty {
                            Text("Record a reflection to populate your map.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        } else {
                            WrapTags(tags: recent)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Emotion Atlas")
            .refreshable {
                await model.refreshEntries()
            }
        }
    }

    private func recentTags(from entries: [APIEntry]) -> [String] {
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let filtered = entries.filter { entry in
            guard let date = DateFormatter.localDate.date(from: entry.localDate) else {
                return false
            }
            return date >= sevenDaysAgo
        }
        let tags = filtered.flatMap { $0.insight?.moodTags ?? [] }
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if !ordered.contains(canonical) {
                ordered.append(canonical)
            }
            if ordered.count >= 24 {
                break
            }
        }
        return ordered
    }
}

struct JournalView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Dot Stream")
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

private enum EmotionAtlasMode: String, CaseIterable, Identifiable {
    case recent
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .recent:
            return "Your Tags"
        case .all:
            return "All Tags"
        }
    }
}

struct EmotionAtlasView: View {
    let entries: [APIEntry]
    var height: CGFloat = 440

    @State private var mode: EmotionAtlasMode = .recent

    private var tagCounts: [String: Int] {
        var counts: [String: Int] = [:]
        for entry in entries {
            guard let tags = entry.insight?.moodTags else { continue }
            for raw in tags {
                let canonical = EmotionTaxonomy.canonicalTag(for: raw)
                counts[canonical, default: 0] += 1
            }
        }
        return counts
    }

    private var hasRecentTags: Bool {
        !tagCounts.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Pleasantness × Energy")
                        .font(.headline)
                    Text("Top right is high-energy pleasant. Bottom left is low-energy unpleasant.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            if hasRecentTags {
                Picker("Scope", selection: $mode) {
                    ForEach(EmotionAtlasMode.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            }

            EmotionAtlasPlane(
                allTags: EmotionTaxonomy.tags,
                tagCounts: tagCounts,
                mode: hasRecentTags ? mode : .all
            )
            .frame(height: height)
            .clipped()

            Text("Inspired by How We Feel's emotional axes. Colors come from each tag's position on the map.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct EmotionAtlasPlane: View {
    let allTags: [EmotionTagDefinition]
    let tagCounts: [String: Int]
    let mode: EmotionAtlasMode

    private var visibleTags: [EmotionTagDefinition] {
        switch mode {
        case .all:
            return allTags
        case .recent:
            let filtered = allTags.filter { tagCounts[$0.id, default: 0] > 0 }
            return filtered.isEmpty ? allTags : filtered
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            let plotInset: CGFloat = 28
            let plotRect = CGRect(
                x: plotInset,
                y: plotInset,
                width: max(1, size.width - (plotInset * 2)),
                height: max(1, size.height - (plotInset * 2))
            )

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.72, green: 0.20, blue: 0.27),
                                Color(red: 0.95, green: 0.77, blue: 0.35),
                                Color(red: 0.43, green: 0.73, blue: 0.58),
                                Color(red: 0.21, green: 0.45, blue: 0.61),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color.black.opacity(0.28))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.white.opacity(0.18), lineWidth: 1)
                    )

                gridLines(in: plotRect)
                    .stroke(Color.white.opacity(0.14), lineWidth: 0.8)

                centerAxes(in: plotRect)
                    .stroke(Color.white.opacity(0.28), lineWidth: 1.2)

                ForEach(visibleTags) { tag in
                    let count = tagCounts[tag.id, default: 0]
                    let isRecent = count > 0
                    let point = position(for: tag, in: plotRect)
                    EmotionTagNode(
                        tag: tag,
                        count: count,
                        emphasized: isRecent,
                        isDimmed: mode == .all && !isRecent
                    )
                    .position(point)
                }
            }
            .overlay(alignment: .top) {
                Text("High Energy")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .padding(.top, 6)
            }
            .overlay(alignment: .bottom) {
                Text("Low Energy")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .padding(.bottom, 6)
            }
            .overlay(alignment: .leading) {
                Text("Unpleasant")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .rotationEffect(.degrees(-90))
                    .padding(.leading, -8)
            }
            .overlay(alignment: .trailing) {
                Text("Pleasant")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .rotationEffect(.degrees(90))
                    .padding(.trailing, -8)
            }
        }
    }

    private func gridLines(in rect: CGRect) -> Path {
        var path = Path()
        let steps = 4
        for step in 1..<steps {
            let ratio = CGFloat(step) / CGFloat(steps)
            let x = rect.minX + (rect.width * ratio)
            let y = rect.minY + (rect.height * ratio)
            path.move(to: CGPoint(x: x, y: rect.minY))
            path.addLine(to: CGPoint(x: x, y: rect.maxY))
            path.move(to: CGPoint(x: rect.minX, y: y))
            path.addLine(to: CGPoint(x: rect.maxX, y: y))
        }
        return path
    }

    private func centerAxes(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        return path
    }

    private func position(for tag: EmotionTagDefinition, in rect: CGRect) -> CGPoint {
        let normalizedX = CGFloat((tag.pleasantness + 1.0) / 2.0)
        let normalizedY = CGFloat(1.0 - ((tag.energy + 1.0) / 2.0))
        let baseX = rect.minX + (rect.width * normalizedX)
        let baseY = rect.minY + (rect.height * normalizedY)
        let jitter = jitterOffset(for: tag.id)
        return CGPoint(
            x: min(rect.maxX, max(rect.minX, baseX + jitter.width)),
            y: min(rect.maxY, max(rect.minY, baseY + jitter.height))
        )
    }

    private func jitterOffset(for text: String) -> CGSize {
        var hash: UInt64 = 1469598103934665603
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        let xRaw = Double(hash % 10_000) / 10_000.0
        let yRaw = Double((hash / 10_000) % 10_000) / 10_000.0
        let maxJitter: CGFloat = 10
        return CGSize(
            width: CGFloat((xRaw * 2.0) - 1.0) * maxJitter,
            height: CGFloat((yRaw * 2.0) - 1.0) * maxJitter
        )
    }
}

private struct EmotionTagNode: View {
    let tag: EmotionTagDefinition
    let count: Int
    let emphasized: Bool
    let isDimmed: Bool

    var body: some View {
        let color = EmotionPalette.color(forTag: tag.id)
        let label = EmotionTaxonomy.displayName(for: tag.id)
        let annotated = count > 1 ? "\(label) ×\(count)" : label

        Text(annotated)
            .font(.system(size: emphasized ? 10.5 : 9.5, weight: emphasized ? .bold : .medium, design: .rounded))
            .foregroundStyle(Color.white.opacity(isDimmed ? 0.7 : 0.98))
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(
                Capsule()
                    .fill(color.opacity(isDimmed ? 0.24 : 0.82))
            )
            .overlay(
                Capsule()
                    .stroke(color.opacity(isDimmed ? 0.35 : 0.95), lineWidth: 0.8)
            )
            .scaleEffect(emphasized ? 1.05 : 1.0)
            .shadow(color: color.opacity(0.22), radius: emphasized ? 6 : 2, x: 0, y: 1)
            .opacity(isDimmed ? 0.45 : 1.0)
    }
}

struct MoodHeatmap: View {
    let entries: [APIEntry]

    private let columns = [GridItem(.adaptive(minimum: 18), spacing: 8)]

    var body: some View {
        let ordered = entries.sorted { lhs, rhs in
            if lhs.localDate != rhs.localDate {
                return lhs.localDate < rhs.localDate
            }
            return lhs.createdAt < rhs.createdAt
        }

        LazyVGrid(columns: columns, spacing: 5) {
            ForEach(ordered) { entry in
                EmotionMixedCircle(
                    tags: entry.insight?.moodTags ?? [],
                    diameter: 16,
                    borderOpacity: 0.24
                )
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct EntryCard: View {
    let entry: APIEntry

    private var statusLabel: String {
        entry.status
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    var body: some View {
        let tags = entry.insight?.moodTags ?? []
        let timeLabel = entry.createdAtTimeLabel

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                EmotionMixedCircle(
                    tags: tags,
                    diameter: 18,
                    borderOpacity: 0.28
                )
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.displayTitle)
                        .font(.headline)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        Text(entry.localDate)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let timeLabel {
                            EntryTimeBadge(timeLabel: timeLabel)
                        }
                    }
                }
                Spacer()
                Text(statusLabel)
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

private struct EntryTimeBadge: View {
    let timeLabel: String

    var body: some View {
        Text(timeLabel)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Color.secondary.opacity(0.14), in: Capsule())
    }
}

struct EntryDetailView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let entry: APIEntry

    @State private var currentEntry: APIEntry
    @StateObject private var audioPlayback = AudioPlaybackController()
    @State private var showDeleteConfirmation = false
    @State private var showTagEditor = false
    @State private var showTitleEditor = false
    @State private var isDeletingEntry = false
    @State private var isSavingTags = false
    @State private var isSavingTitle = false
    @State private var isRetranscribing = false

    init(entry: APIEntry) {
        self.entry = entry
        _currentEntry = State(initialValue: entry)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(currentEntry.displayTitle)
                        .font(.title2.bold())
                        .lineLimit(2)
                        .minimumScaleFactor(0.85)

                    HStack(spacing: 8) {
                        Text(currentEntry.localDate)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        if let timeLabel = currentEntry.createdAtTimeLabel {
                            EntryTimeBadge(timeLabel: timeLabel)
                        }
                    }
                }

                if let insight = currentEntry.insight {
                    VStack(alignment: .leading, spacing: 10) {
                        EmotionMixedCircle(
                            tags: insight.moodTags,
                            diameter: 186,
                            borderOpacity: 0.34
                        )
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 4)

                        HStack(spacing: 8) {
                            Text("Tags")
                                .font(.headline)
                            Spacer()
                            Button("Edit Tags") {
                                showTagEditor = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(isSavingTags || isDeletingEntry)
                        }
                        WrapTags(tags: insight.moodTags)

                        if isSavingTags {
                            ProgressView("Saving tags...")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if let transcript = currentEntry.transcript {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Transcript")
                            .font(.headline)
                        Text(transcript.text)
                            .font(.body)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Transcript")
                            .font(.headline)
                        Text("No transcript yet. Use Retranscribe to process this audio note.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Audio")
                        .font(.headline)

                    HStack(spacing: 10) {
                        Button(audioPlayback.isPlaying ? "Pause" : "Play") {
                            Task {
                                await togglePlayback()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(audioPlayback.isLoading)
                    }

                    Button(isRetranscribing ? "Retranscribing..." : "Retranscribe") {
                        Task {
                            await retranscribeEntry()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRetranscribing || isDeletingEntry)

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
        .navigationTitle(currentEntry.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button {
                    showTitleEditor = true
                } label: {
                    Image(systemName: "pencil")
                }
                .disabled(isDeletingEntry || isSavingTitle)

                Button(role: .destructive) {
                    showDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                }
                .disabled(isDeletingEntry)
            }
        }
        .alert("Delete reflection?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                Task { await deleteEntry() }
            }
            .disabled(isDeletingEntry)
        } message: {
            Text("This removes the local entry, transcript, audio, and social dot for this day.")
        }
        .sheet(isPresented: $showTagEditor) {
            if let insight = currentEntry.insight {
                EntryTagEditorSheet(
                    initialTags: insight.moodTags,
                    isSaving: isSavingTags,
                    onSave: { tags in
                        Task {
                            await saveTags(tags)
                        }
                    }
                )
            } else {
                VStack(spacing: 12) {
                    Text("No tags available for this entry.")
                        .font(.headline)
                    Button("Close") {
                        showTagEditor = false
                    }
                    .buttonStyle(.bordered)
                }
                .padding(24)
            }
        }
        .sheet(isPresented: $showTitleEditor) {
            EntryTitleEditorSheet(
                initialTitle: currentEntry.displayTitle,
                isSaving: isSavingTitle,
                onSave: { newTitle in
                    Task {
                        await saveTitle(newTitle)
                    }
                }
            )
        }
        .task {
            if let refreshed = model.entries.first(where: { $0.id == entry.id }) {
                currentEntry = refreshed
            }
            await loadAudio(forceReload: false)
        }
        .onDisappear {
            audioPlayback.stop()
        }
    }

    private func loadAudio(forceReload: Bool) async {
        do {
            try await audioPlayback.load(
                fetchAudio: { try await model.fetchAudio(entryID: currentEntry.id) },
                forceReload: forceReload
            )
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private func togglePlayback() async {
        if !audioPlayback.isReady {
            await loadAudio(forceReload: true)
        }
        guard audioPlayback.isReady else { return }
        audioPlayback.togglePlayback()
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded()))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }

    private func deleteEntry() async {
        guard !isDeletingEntry else {
            return
        }
        isDeletingEntry = true
        await model.deleteEntry(entryID: currentEntry.id)
        isDeletingEntry = false
        if !model.entries.contains(where: { $0.id == currentEntry.id }) {
            dismiss()
        }
    }

    private func saveTags(_ tags: [String]) async {
        guard !isSavingTags else { return }
        isSavingTags = true
        if let updated = await model.updateEntryTags(
            entryID: currentEntry.id,
            moodTags: tags
        ) {
            currentEntry = updated
            showTagEditor = false
        }
        isSavingTags = false
    }

    private func saveTitle(_ title: String) async {
        guard !isSavingTitle else { return }
        isSavingTitle = true
        if let updated = await model.updateEntryTitle(
            entryID: currentEntry.id,
            title: title
        ) {
            currentEntry = updated
            showTitleEditor = false
        }
        isSavingTitle = false
    }

    private func retranscribeEntry() async {
        guard !isRetranscribing else { return }
        isRetranscribing = true
        if let updated = await model.retranscribeEntry(entryID: currentEntry.id) {
            currentEntry = updated
        }
        isRetranscribing = false
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

private struct EntryTitleEditorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let isSaving: Bool
    let onSave: (String) -> Void

    @State private var titleValue: String

    init(
        initialTitle: String,
        isSaving: Bool,
        onSave: @escaping (String) -> Void
    ) {
        self.isSaving = isSaving
        self.onSave = onSave
        _titleValue = State(initialValue: APIEntry.sanitizeTitle(initialTitle))
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                Text("Title")
                    .font(.headline)
                TextField("Entry", text: $titleValue, axis: .vertical)
                    .textInputAutocapitalization(.sentences)
                    .autocorrectionDisabled(false)
                    .lineLimit(2)
                    .padding(12)
                    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                Text("Shown on the journal timeline and this entry.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .padding(16)
            .navigationTitle("Edit Title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(isSaving)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onSave(titleValue)
                    }
                    .disabled(isSaving)
                }
            }
        }
    }
}

private struct EntryTagEditorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let isSaving: Bool
    let onSave: ([String]) -> Void

    @State private var selectedTags: [String]

    init(
        initialTags: [String],
        isSaving: Bool,
        onSave: @escaping ([String]) -> Void
    ) {
        let normalized = EntryTagEditorSheet.sanitized(initialTags)
        self.isSaving = isSaving
        self.onSave = onSave
        _selectedTags = State(initialValue: normalized)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Select up to 4 tags")
                        .font(.headline)
                    Text("\(selectedTags.count)/4 selected")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 128), spacing: 8)],
                        spacing: 8
                    ) {
                        ForEach(EmotionTaxonomy.tags) { definition in
                            Button {
                                toggle(definition.id)
                            } label: {
                                SelectableTagChip(
                                    tag: definition.id,
                                    isSelected: selectedTags.contains(definition.id)
                                )
                            }
                            .buttonStyle(.plain)
                            .disabled(isSaving)
                        }
                    }

                    Text("Tags update your journal entry and social dot for this day if the entry was shared.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.top, 4)
                }
                .padding(16)
            }
            .navigationTitle("Edit Tags")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(isSaving)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onSave(selectedTags)
                    }
                    .disabled(selectedTags.isEmpty || isSaving)
                }
            }
        }
    }

    private func toggle(_ tag: String) {
        let canonical = EmotionTaxonomy.canonicalTag(for: tag)
        if let existingIndex = selectedTags.firstIndex(of: canonical) {
            selectedTags.remove(at: existingIndex)
            return
        }
        guard selectedTags.count < 4 else {
            return
        }
        selectedTags.append(canonical)
    }

    private static func sanitized(_ tags: [String]) -> [String] {
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
}

private struct SelectableTagChip: View {
    let tag: String
    let isSelected: Bool

    var body: some View {
        let color = EmotionPalette.color(forTag: tag)

        HStack(spacing: 6) {
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .font(.caption.weight(.semibold))
            Text(EmotionTaxonomy.displayName(for: tag))
                .font(.caption.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .foregroundStyle(isSelected ? Color.primary : Color.primary.opacity(0.86))
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [
                    color.opacity(isSelected ? 0.44 : 0.26),
                    color.opacity(isSelected ? 0.22 : 0.14),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: Capsule()
        )
        .overlay(
            Capsule()
                .stroke(color.opacity(isSelected ? 0.88 : 0.42), lineWidth: isSelected ? 1.3 : 1.0)
        )
    }
}

enum EmotionPalette {
    static func color(forTag tag: String) -> Color {
        if let definition = EmotionTaxonomy.definition(for: tag) {
            return color(
                pleasantness: definition.pleasantness,
                energy: definition.energy
            )
        }

        var hash: UInt64 = 5381
        for byte in EmotionTaxonomy.normalize(tag).utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        let hue = Double(hash % 360) / 360.0
        return Color(hue: hue, saturation: 0.55, brightness: 0.82)
    }

    private static func color(pleasantness: Double, energy: Double) -> Color {
        let clampedPleasantness = max(-1.0, min(1.0, pleasantness))
        let clampedEnergy = max(-1.0, min(1.0, energy))
        let hue = 0.01 + ((clampedPleasantness + 1.0) / 2.0) * 0.47
        let saturation = min(
            0.92,
            0.45
            + abs(clampedPleasantness) * 0.28
            + max(0.0, clampedEnergy) * 0.18
        )
        let brightness = min(0.96, 0.58 + ((clampedEnergy + 1.0) / 2.0) * 0.28)
        return Color(hue: hue, saturation: saturation, brightness: brightness)
    }
}

struct TagChip: View {
    let tag: String

    private var label: String {
        EmotionTaxonomy.displayName(for: tag)
    }

    var body: some View {
        let color = EmotionPalette.color(forTag: tag)

        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                LinearGradient(
                    colors: [
                        color.opacity(0.38),
                        color.opacity(0.16),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: Capsule()
            )
            .overlay(
                Capsule()
                    .stroke(color.opacity(0.48), lineWidth: 1)
            )
    }
}

// MARK: - Social

struct SocialView: View {
    @EnvironmentObject private var model: AppModel
    @State private var selectedDotID: String?

    private let columns = [GridItem(.adaptive(minimum: 66), spacing: 18)]

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.opacity(0.02).ignoresSafeArea()

                if model.socialDots.isEmpty {
                    Text("No friend dots yet.\nAdd friends in Settings.")
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                        .padding()
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 20) {
                            ForEach(model.socialDots) { dot in
                                VStack(spacing: 6) {
                                    Button {
                                        withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                                            if selectedDotID == dot.id {
                                                selectedDotID = nil
                                            } else {
                                                selectedDotID = dot.id
                                            }
                                        }
                                    } label: {
                                        EmotionMixedCircle(
                                            tags: dot.dotTags ?? [],
                                            fallbackHex: dot.dotColor,
                                            diameter: 46,
                                            borderOpacity: 0.38
                                        )
                                        .frame(width: 58, height: 58)
                                    }
                                    .buttonStyle(.plain)

                                    Text(dot.label ?? "@\(dot.userId.prefix(6))")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.75)
                                }
                                .frame(width: 84)
                                .overlay(alignment: .bottom) {
                                    if selectedDotID == dot.id {
                                        SocialDotTagBubble(tags: dot.dotTags ?? [])
                                            .offset(y: 92)
                                            .transition(.opacity.combined(with: .scale(scale: 0.92)))
                                    }
                                }
                                .zIndex(selectedDotID == dot.id ? 2 : 0)
                            }
                        }
                        .padding(.horizontal, 26)
                        .padding(.top, 24)
                        .padding(.bottom, 132)
                        .frame(maxWidth: .infinity, alignment: .center)
                    }
                }
            }
            .navigationTitle("Social")
            .refreshable {
                await model.refreshSocialDots()
            }
            .onChange(of: model.socialDots) { _, _ in
                if let selectedDotID, !model.socialDots.contains(where: { $0.id == selectedDotID }) {
                    self.selectedDotID = nil
                }
            }
        }
    }
}

private struct SocialDotTagBubble: View {
    let tags: [String]

    private var normalizedTags: [String] {
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

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Dot makeup")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)

            if normalizedTags.isEmpty {
                Text("No tags shared")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 70), spacing: 6)],
                    spacing: 6
                ) {
                    ForEach(normalizedTags, id: \.self) { tag in
                        TagChip(tag: tag)
                    }
                }
            }
        }
        .padding(10)
        .frame(width: 184, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.16), radius: 10, x: 0, y: 4)
    }
}

struct ReminderScheduleEditor: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Days")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(ReminderWeekday.ordered) { day in
                        let selected = model.isReminderWeekdaySelected(day)
                        Button(day.shortTitle) {
                            model.toggleReminderWeekday(day)
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(
                            selected
                            ? Color.teal.opacity(0.36)
                            : Color.secondary.opacity(0.16),
                            in: Capsule()
                        )
                        .overlay(
                            Capsule()
                                .stroke(
                                    selected ? Color.teal.opacity(0.75) : Color.white.opacity(0.14),
                                    lineWidth: 1
                                )
                        )
                    }
                }
            }

            ForEach(model.selectedReminderDays()) { day in
                HStack {
                    Text(day.title)
                        .font(.subheadline)
                    Spacer()
                    DatePicker(
                        "",
                        selection: Binding(
                            get: { model.reminderTime(for: day) },
                            set: { model.setReminderTime($0, for: day) }
                        ),
                        displayedComponents: .hourAndMinute
                    )
                    .labelsHidden()
                }
            }
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var acceptInviteToken: String = ""
    @State private var clipboardStatus: String?
    @State private var feedbackKind: FeedbackKind = .idea
    @State private var feedbackMessage: String = ""
    @State private var feedbackStatus: String?
    @State private var isSubmittingFeedback = false
    @State private var feedbackToastMessage: String?
    @FocusState private var focusedInput: FocusedInput?

    private enum FocusedInput: Hashable {
        case feedbackMessage
    }

    private enum FeedbackKind: String, CaseIterable, Identifiable {
        case idea
        case bug

        var id: String { rawValue }

        var title: String {
            switch self {
            case .idea:
                return "Idea"
            case .bug:
                return "Bug"
            }
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Display name", text: $model.displayName)
                    Text("Anonymous handle: @\(model.anonymousHandle)")
                }

                Section("Check-In") {
                    DatePicker("Default time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
                    ReminderScheduleEditor()
                    Button("Save Reminder Schedule") {
                        Task {
                            await model.configureDailyReminder()
                        }
                    }
                    if let reminderStatus = model.reminderStatus {
                        Text(reminderStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Integrations") {
                    Text("HealthKit (V2 planned): sleep, HRV, and activity.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("On-Device AI") {
                    Button(model.isPreparingLiquidModel ? "Preparing Model..." : "Redownload Liquid Model") {
                        model.redownloadLiquidModel()
                    }
                    .disabled(model.isPreparingLiquidModel)

                    if model.isPreparingLiquidModel {
                        ProgressView(value: model.liquidModelPreparationProgress)
                    }

                    Text("Use this if local model files are corrupted or you want a fresh download.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Social") {
                    Button("Create Invite Link") {
                        Task {
                            await model.createInvite()
                        }
                    }

                    if let inviteToken = model.inviteToken {
                        HStack(alignment: .top) {
                            Text(inviteToken)
                                .font(.footnote.monospaced())
                                .textSelection(.enabled)
                            Spacer()
                            Button("Copy Token") {
                                copyToClipboard(inviteToken, label: "invite token")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    if let clipboardStatus {
                        Text(clipboardStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    TextField("Paste invite link or token", text: $acceptInviteToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button("Accept Invite") {
                        Task {
                            await model.acceptInvite(token: acceptInviteToken)
                            acceptInviteToken = ""
                        }
                    }
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

                Section("Send Idea / Bug Report") {
                    Picker("Type", selection: $feedbackKind) {
                        ForEach(FeedbackKind.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)

                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $feedbackMessage)
                            .frame(minHeight: 110)
                            .focused($focusedInput, equals: .feedbackMessage)
                        if feedbackMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text("Tell us what happened or what you'd like to see.")
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                        }
                    }

                    Button(isSubmittingFeedback ? "Sending..." : "Send Report") {
                        Task {
                            await submitFeedback()
                        }
                    }
                    .disabled(
                        isSubmittingFeedback
                            || feedbackMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    )

                    if let feedbackStatus {
                        Text(feedbackStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("About") {
                    Text("Made by Zubin @ lowercaseLabs")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        focusedInput = nil
                    }
                }
            }
            .overlay(alignment: .bottom) {
                if let feedbackToastMessage {
                    Text(feedbackToastMessage)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.black.opacity(0.82))
                        )
                        .padding(.bottom, 22)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
    }

    private func copyToClipboard(_ value: String, label: String) {
        UIPasteboard.general.string = value
        clipboardStatus = "Copied \(label)."
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
            clipboardStatus = nil
        }
    }

    private func submitFeedback() async {
        guard !isSubmittingFeedback else { return }
        focusedInput = nil
        isSubmittingFeedback = true
        let succeeded = await model.submitFeedback(kind: feedbackKind.rawValue, message: feedbackMessage)
        if succeeded {
            feedbackMessage = ""
            feedbackStatus = "Submitted."
            showFeedbackToast("Report submitted")
        }
        isSubmittingFeedback = false
    }

    private func showFeedbackToast(_ message: String) {
        withAnimation(.easeInOut(duration: 0.2)) {
            feedbackToastMessage = message
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation(.easeInOut(duration: 0.2)) {
                feedbackToastMessage = nil
            }
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

    static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "h:mm a"
        return formatter
    }()
}

#Preview {
    ContentView()
}
