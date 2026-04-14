import AVFoundation
import CryptoKit
import Foundation
import os

#if canImport(LeapSDK)
import LeapSDK
#endif

#if canImport(LeapModelDownloader)
import LeapModelDownloader
#endif

private enum LiquidInsightsConfig {
    // Update this tuple to test different hosted Liquid model variants.
    static let modelTuple: (model: String, quantization: String) = ("LFM2.5-1.2B-Instruct", "Q5_K_M")
    static let defaultSequenceLength: UInt32 = 1024
    static let maxOutputTokens: UInt32 = 256
}

enum LocalInsightPipelineVersion: String {
    case v0TranscriptStructured = "v0_transcript_structured"
}

private struct LocalInsightExtractionResult {
    let insight: APIInsight
    let title: String
    let provider: String
    let pipelineVersion: LocalInsightPipelineVersion
    let latencyMilliseconds: Int
    let modelID: String?
    let tokensPerSecond: Double?
    let fallbackReason: String?
}

private struct LiquidStructuredOutput {
    let title: String
    let moodTags: [String]
    let modelID: String?
    let tokensPerSecond: Double?
}

private struct LocalTranscriptionResult {
    let text: String
    let strategy: String
    let audioDurationSeconds: Double?
    let coverageSeconds: Double?
    let chunkFallbackUsed: Bool
}

enum LocalReflectionAnalyzer {
    private static let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.lowercaseLabs.theVoid", category: "liquid-insights")

    private static func verboseLoggingEnabled() -> Bool {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: "thevoid.liquid.verboseLogs") == nil {
            return true
        }
        return defaults.bool(forKey: "thevoid.liquid.verboseLogs")
    }

    private static func debugLog(_ message: String) {
        guard verboseLoggingEnabled() else { return }
        logger.debug("\(message, privacy: .public)")
    }

    private static func errorLog(_ message: String) {
        guard verboseLoggingEnabled() else { return }
        logger.error("\(message, privacy: .public)")
    }

    @discardableResult
    static func preloadLiquidModelIfNeeded(enabled: Bool) async -> String? {
        guard enabled else { return nil }
        debugLog("preloadLiquidModelIfNeeded requested")
        let descriptor = try? await LiquidInsightsRuntime.shared.preload()
        debugLog("preloadLiquidModelIfNeeded result descriptor=\(descriptor ?? "nil")")
        return descriptor
    }

    @discardableResult
    static func prepareLiquidModel(
        forceRedownload: Bool = false,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws -> String {
        debugLog("prepareLiquidModel requested forceRedownload=\(forceRedownload)")
        let descriptor = try await LiquidInsightsRuntime.shared.prepareModel(
            forceRedownload: forceRedownload,
            progressHandler: progressHandler
        )
        debugLog("prepareLiquidModel completed descriptor=\(descriptor)")
        return descriptor
    }

    static func cancelLiquidModelPreparation() async {
        debugLog("cancelLiquidModelPreparation requested")
        await LiquidInsightsRuntime.shared.cancelActiveDownloads()
    }

    static func analyze(
        audioURL: URL,
        durationSeconds: Int,
        useLiquidInsights: Bool,
        overrideTranscriptText: String? = nil
    ) async -> (APITranscript?, APIInsight?, String?) {
        debugLog("analyze start duration=\(durationSeconds) useLiquidInsights=\(useLiquidInsights) override=\(overrideTranscriptText != nil)")
        // Warm model load while speech transcription runs so first-run latency is hidden when possible.
        if useLiquidInsights {
            Task {
                _ = await preloadLiquidModelIfNeeded(enabled: true)
            }
        }

        let transcription: LocalTranscriptionResult
        if let override = overrideTranscriptText {
            let audioDuration = audioDurationSeconds(for: audioURL)
            transcription = LocalTranscriptionResult(
                text: override,
                strategy: "whisperkit_tiny_en_edited",
                audioDurationSeconds: audioDuration,
                coverageSeconds: audioDuration,
                chunkFallbackUsed: false
            )
        } else {
            do {
                transcription = try await transcribeOnDevice(audioURL: audioURL)
            } catch {
                errorLog("analyze transcription failed reason=\(describeErrorChain(error))")
                return (nil, nil, nil)
            }
        }

        let transcriptText = transcription.text
        let normalized = transcriptText.trimmingCharacters(in: .whitespacesAndNewlines)
        var metadata: [String: JSONValue] = [
            "provider": .string("whisperkit"),
            "duration_seconds": .int(max(1, min(durationSeconds, 150))),
            "transcription_strategy": .string(transcription.strategy),
        ]

        if let audioDurationSeconds = transcription.audioDurationSeconds {
            metadata["audio_duration_seconds"] = .double(audioDurationSeconds)
        }
        if let coverageSeconds = transcription.coverageSeconds {
            metadata["transcription_coverage_seconds"] = .double(coverageSeconds)
        }

        let transcript = APITranscript(
            text: normalized.isEmpty ? "No speech detected in reflection." : normalized,
            providerMetadata: metadata
        )

        guard useLiquidInsights else {
            metadata["insight_mode_requested"] = .string("disabled")
            let transcriptOnly = APITranscript(
                text: transcript.text,
                providerMetadata: metadata
            )
            return (transcriptOnly, nil, nil)
        }

        let extracted = await extractInsight(
            from: normalized,
            useLiquidInsights: true
        )
        debugLog(
            "analyze extraction complete provider=\(extracted.provider) pipeline=\(extracted.pipelineVersion.rawValue) latencyMs=\(extracted.latencyMilliseconds) model=\(extracted.modelID ?? "nil")"
        )

        metadata["insight_provider"] = .string(extracted.provider)
        metadata["insight_pipeline"] = .string(extracted.pipelineVersion.rawValue)
        metadata["insight_mode_requested"] = .string("liquid_insights")
        metadata["insight_latency_ms"] = .int(extracted.latencyMilliseconds)
        if let modelID = extracted.modelID {
            metadata["insight_model"] = .string(modelID)
        }
        if let tokensPerSecond = extracted.tokensPerSecond {
            metadata["insight_tokens_per_second"] = .double(tokensPerSecond)
        }
        if let fallbackReason = extracted.fallbackReason {
            metadata["insight_fallback_reason"] = .string(fallbackReason)
        }
        metadata["insight_title"] = .string(extracted.title)

        let transcriptWithInsightMetadata = APITranscript(
            text: transcript.text,
            providerMetadata: metadata
        )
        return (transcriptWithInsightMetadata, extracted.insight, extracted.title)
    }

    private static func transcribeOnDevice(audioURL: URL) async throws -> LocalTranscriptionResult {
        let audioDuration = audioDurationSeconds(for: audioURL)
        let text = try await WhisperTranscriptionRuntime.shared.transcribe(audioURL: audioURL)
        return LocalTranscriptionResult(
            text: text,
            strategy: "whisperkit_tiny_en",
            audioDurationSeconds: audioDuration,
            coverageSeconds: audioDuration,
            chunkFallbackUsed: false
        )
    }

    private static func normalizeTranscriptSpacing(_ text: String) -> String {
        let collapsed = text.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )
        let punctuationFixed = collapsed.replacingOccurrences(
            of: "\\s+([,.;:!?])",
            with: "$1",
            options: .regularExpression
        )
        return punctuationFixed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func audioDurationSeconds(for url: URL) -> Double? {
        let seconds = AVURLAsset(url: url).duration.seconds
        guard seconds.isFinite, seconds > 0 else {
            return nil
        }
        return seconds
    }

    private static func formatSeconds(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite else {
            return "nil"
        }
        return String(format: "%.2f", seconds)
    }

    private static func describeErrorChain(_ error: Error) -> String {
        var parts: [String] = []
        var current: NSError? = error as NSError

        while let nsError = current {
            parts.append(nsError.localizedDescription)
            current = nsError.userInfo[NSUnderlyingErrorKey] as? NSError
        }

        if parts.isEmpty {
            return String(describing: error)
        }
        return parts.joined(separator: " | ")
    }

    private static func extractInsight(
        from transcriptText: String,
        useLiquidInsights: Bool
    ) async -> LocalInsightExtractionResult {
        guard useLiquidInsights else {
            return extractInsightWithKeywords(from: transcriptText, provider: "keyword_match", fallbackReason: nil)
        }

        do {
            return try await extractInsightV0Liquid(from: transcriptText)
        } catch {
            let reason = String(describing: error)
            errorLog("extractInsight liquid path failed, falling back to keywords reason=\(reason)")
            return extractInsightWithKeywords(from: transcriptText, provider: "keyword_fallback", fallbackReason: reason)
        }
    }

    // V0: transcript -> structured extraction with Liquid model.
    private static func extractInsightV0Liquid(from transcriptText: String) async throws -> LocalInsightExtractionResult {
        let startNanos = DispatchTime.now().uptimeNanoseconds
        let output = try await LiquidInsightsRuntime.shared.extractV0(
            transcriptText: transcriptText,
            allowedTags: EmotionTaxonomy.tags.map(\.id)
        )

        let lowered = transcriptText.lowercased()
        var tags = sanitizeModelTags(output.moodTags)
        if tags.isEmpty {
            tags = extractMoodTags(from: lowered)
        }
        let title = normalizedGeneratedTitle(
            output.title,
            fallbackTranscript: transcriptText
        )

        let insight = APIInsight(
            moodScore: EmotionTaxonomy.moodScore(for: tags),
            moodTags: tags,
            themes: [],
            signals: signals(from: lowered, tags: tags),
            safetyFlags: nil
        )

        let elapsed = elapsedMilliseconds(since: startNanos)
        return LocalInsightExtractionResult(
            insight: insight,
            title: title,
            provider: "liquid_structured_v0",
            pipelineVersion: .v0TranscriptStructured,
            latencyMilliseconds: elapsed,
            modelID: output.modelID,
            tokensPerSecond: output.tokensPerSecond,
            fallbackReason: nil
        )
    }

    private static func extractInsightWithKeywords(
        from transcriptText: String,
        provider: String,
        fallbackReason: String?
    ) -> LocalInsightExtractionResult {
        let startNanos = DispatchTime.now().uptimeNanoseconds
        let lowered = transcriptText.lowercased()
        let tags = extractMoodTags(from: lowered)
        let title = fallbackTitle(from: transcriptText)
        let insight = APIInsight(
            moodScore: EmotionTaxonomy.moodScore(for: tags),
            moodTags: tags,
            themes: [],
            signals: signals(from: lowered, tags: tags),
            safetyFlags: nil
        )

        return LocalInsightExtractionResult(
            insight: insight,
            title: title,
            provider: provider,
            pipelineVersion: .v0TranscriptStructured,
            latencyMilliseconds: elapsedMilliseconds(since: startNanos),
            modelID: nil,
            tokensPerSecond: nil,
            fallbackReason: fallbackReason
        )
    }

    private static func elapsedMilliseconds(since startNanos: UInt64) -> Int {
        let endNanos = DispatchTime.now().uptimeNanoseconds
        let delta = endNanos >= startNanos ? (endNanos - startNanos) : 0
        return Int(delta / 1_000_000)
    }

    private static func containsPhrase(_ text: String, phrase: String) -> Bool {
        if phrase.contains(" ") {
            return text.contains(phrase)
        }
        let pattern = "\\b\(NSRegularExpression.escapedPattern(for: phrase))\\b"
        return text.range(of: pattern, options: .regularExpression) != nil
    }

    private static func sanitizeModelTags(_ rawTags: [String]) -> [String] {
        var tags: [String] = []
        for raw in rawTags {
            guard let definition = EmotionTaxonomy.definition(for: raw) else {
                continue
            }
            if tags.contains(definition.id) {
                continue
            }
            tags.append(definition.id)
            if tags.count >= 4 {
                break
            }
        }
        return tags
    }

    private static func extractMoodTags(from lowered: String) -> [String] {
        EmotionTaxonomy.matchedTags(in: lowered, maxCount: 4) { phrase in
            containsPhrase(lowered, phrase: phrase)
        }
    }

    private static func signals(from lowered: String, tags: [String]) -> [String: Double] {
        let pleasantness = EmotionTaxonomy.averagePleasantness(for: tags)
        let averageEnergy = EmotionTaxonomy.averageEnergy(for: tags)
        let stressFromWords: Double = ["stress", "stressed", "anxious", "overwhelmed"]
            .contains { containsPhrase(lowered, phrase: $0) } ? 0.75 : 0.45
        let stressFromPosition = max(0.1, min(0.95, (1.0 - ((pleasantness + 1.0) / 2.0))))
        let stress = max(stressFromWords, stressFromPosition)
        let energy = max(0.1, min(0.95, (averageEnergy + 1.0) / 2.0))
        let confidenceFromWords: Double = ["confident", "sure", "ready", "capable"]
            .contains { containsPhrase(lowered, phrase: $0) } ? 0.7 : 0.4
        let confidenceFromPosition = max(0.2, min(0.9, ((pleasantness + 1.0) / 2.0)))
        let confidence = max(confidenceFromWords, confidenceFromPosition)
        return [
            "stress": stress,
            "energy": energy,
            "confidence": confidence,
        ]
    }

    private static func normalizedGeneratedTitle(_ rawTitle: String, fallbackTranscript: String) -> String {
        let extractedWords = titleWords(from: rawTitle)
        guard !extractedWords.isEmpty else {
            return fallbackTitle(from: fallbackTranscript)
        }

        let clamped = Array(extractedWords.prefix(5))
        var outputWords = clamped
        if outputWords.count == 1 {
            outputWords.append("Reflection")
        }
        if outputWords.isEmpty {
            return fallbackTitle(from: fallbackTranscript)
        }
        return formatTitleWords(outputWords)
    }

    private static func fallbackTitle(from transcriptText: String) -> String {
        let words = titleWords(from: transcriptText)
        if words.isEmpty {
            return "Daily Reflection"
        }
        let count = words.count == 1 ? 1 : min(words.count, 5)
        var selected = Array(words.prefix(count))
        if selected.count == 1 {
            selected.append("Reflection")
        }
        return formatTitleWords(selected)
    }

    private static func titleWords(from text: String) -> [String] {
        let cleaned = text.replacingOccurrences(
            of: "[^A-Za-z0-9' ]+",
            with: " ",
            options: .regularExpression
        )
        return cleaned
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func formatTitleWords(_ words: [String]) -> String {
        let normalized = words
            .prefix(5)
            .map { $0.lowercased().capitalized }
        return APIEntry.sanitizeTitle(normalized.joined(separator: " "))
    }
}

private enum LiquidInsightsError: LocalizedError {
    case unavailable
    case emptyTranscript
    case modelOutputMissing

    var errorDescription: String? {
        switch self {
        case .unavailable:
            return "Liquid insights runtime is unavailable in this build."
        case .emptyTranscript:
            return "No transcript text was available for Liquid extraction."
        case .modelOutputMissing:
            return "Liquid model did not return structured output."
        }
    }
}

private actor LiquidInsightsRuntime {
    static let shared = LiquidInsightsRuntime()
    typealias ProgressHandler = @Sendable (Double) -> Void

    private enum DefaultsKeys {
        static let localBundlePath = "thevoid.liquid.localBundlePath"
        static let enableCustomModelSources = "thevoid.liquid.enableCustomModelSources"
        static let modelFileURL = "thevoid.liquid.modelFileURL"
        static let modelManifestURL = "thevoid.liquid.modelManifestURL"
        static let verboseLogs = "thevoid.liquid.verboseLogs"
        static let sequenceLength = "thevoid.liquid.sequenceLength"
        static let maxOutputTokens = "thevoid.liquid.maxOutputTokens"
        static let unloadAfterEachGeneration = "thevoid.liquid.unloadAfterEachGeneration"
    }

    private let defaultModel = LiquidInsightsConfig.modelTuple.model
    private let defaultQuantization = LiquidInsightsConfig.modelTuple.quantization
    private let customModelSourcesEnabledByDefault = false
    private let remoteLoadMaxAttempts = 3
    private let defaultSequenceLength: UInt32 = LiquidInsightsConfig.defaultSequenceLength
    private let defaultMaxOutputTokens: UInt32 = LiquidInsightsConfig.maxOutputTokens
    private let defaultHostedModelBaseURL = "https://thevoid-local.fly.dev/models"

#if canImport(LeapSDK)
    private var modelRunner: ModelRunner?
#endif
    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.lowercaseLabs.theVoid", category: "liquid-insights")
    private var activeModelDescriptor: String?
    private var activeGenerationCount = 0
    private var activeCustomDownloadSession: URLSession?
    private var activeRemoteDownloadTarget: (model: String, quantization: String)?
#if canImport(LeapModelDownloader)
    private var activeModelDownloader: ModelDownloader?
#endif
    private var isLoadingRunner = false
#if canImport(LeapSDK)
    private var runnerLoadWaiters: [CheckedContinuation<ModelRunner, Error>] = []
#endif
    private var generationBusy = false
    private var generationWaiters: [CheckedContinuation<Void, Never>] = []

    private func isVerboseLoggingEnabled() -> Bool {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: DefaultsKeys.verboseLogs) == nil {
            return true
        }
        return defaults.bool(forKey: DefaultsKeys.verboseLogs)
    }

    private func debugLog(_ message: String) {
        guard isVerboseLoggingEnabled() else { return }
        logger.info("\(message, privacy: .public)")
    }

    private func errorLog(_ message: String) {
        guard isVerboseLoggingEnabled() else { return }
        logger.error("\(message, privacy: .public)")
    }

    private func downloadStatusDescription(_ status: ModelDownloader.ModelDownloadStatus) -> String {
        switch status {
        case .notOnLocal:
            return "notOnLocal"
        case .downloaded:
            return "downloaded"
        case .downloadInProgress(let progress):
            let percentage = Int((progress * 100).rounded())
            return "downloadInProgress(\(percentage)%)"
        }
    }

    private func fileDescription(at path: String) -> String {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: path) else {
            return "missing"
        }
        do {
            let attrs = try fileManager.attributesOfItem(atPath: path)
            let size = (attrs[.size] as? NSNumber)?.int64Value ?? -1
            return "exists size=\(size)"
        } catch {
            return "exists size=unknown error=\(error.localizedDescription)"
        }
    }

    private func waitForGenerationTurn() async {
        if !generationBusy {
            generationBusy = true
            return
        }

        await withCheckedContinuation { continuation in
            generationWaiters.append(continuation)
        }
    }

    private func finishGenerationTurn() {
        if generationWaiters.isEmpty {
            generationBusy = false
            return
        }
        let next = generationWaiters.removeFirst()
        next.resume()
    }

#if canImport(LeapSDK)
    private func waitForInFlightRunnerLoad() async throws -> ModelRunner {
        try await withCheckedThrowingContinuation { continuation in
            runnerLoadWaiters.append(continuation)
        }
    }

    private func finishRunnerLoad(with result: Result<ModelRunner, Error>) {
        let waiters = runnerLoadWaiters
        runnerLoadWaiters.removeAll(keepingCapacity: false)
        for waiter in waiters {
            waiter.resume(with: result)
        }
    }
#endif

    private func configuredSequenceLength() -> UInt32 {
        let configured = UserDefaults.standard.integer(forKey: DefaultsKeys.sequenceLength)
        guard configured > 0 else {
            return defaultSequenceLength
        }
        return UInt32(max(512, min(configured, 4096)))
    }

    private func configuredMaxOutputTokens() -> UInt32 {
        let configured = UserDefaults.standard.integer(forKey: DefaultsKeys.maxOutputTokens)
        guard configured > 0 else {
            return defaultMaxOutputTokens
        }
        return UInt32(max(32, min(configured, Int(LiquidInsightsConfig.maxOutputTokens))))
    }

    private func shouldUnloadAfterEachGeneration() -> Bool {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: DefaultsKeys.unloadAfterEachGeneration) == nil {
            return true
        }
        return defaults.bool(forKey: DefaultsKeys.unloadAfterEachGeneration)
    }

    private func customModelSourcesEnabled() -> Bool {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: DefaultsKeys.enableCustomModelSources) == nil {
            return customModelSourcesEnabledByDefault
        }
        return defaults.bool(forKey: DefaultsKeys.enableCustomModelSources)
    }

    func extractV0(
        transcriptText: String,
        allowedTags: [String]
    ) async throws -> LiquidStructuredOutput {
        let trimmed = transcriptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LiquidInsightsError.emptyTranscript
        }

        let systemPrompt = Self.v0SystemPrompt(
            allowedTags: allowedTags
        )

        let requestBody = """
        Transcript:
        \(trimmed)

        Return only JSON matching the schema.
        """

        let result: LiquidStructuredOutput = try await generateStructured(
            systemPrompt: systemPrompt,
            message: ChatMessage(role: .user, content: [.text(requestBody)])
        )
        debugLog(
            "extractV0 result title=\(result.title) moodTags=\(result.moodTags) modelID=\(result.modelID ?? "nil") tokensPerSecond=\(result.tokensPerSecond?.description ?? "nil")"
        )
        return result
    }

    private static func v0SystemPrompt(
        allowedTags: [String]
    ) -> String {
        """
        You are an analysis engine for private reflection transcripts.
        Return strict JSON only.

        Rules:
        - Return exactly two keys: title and moodTags.
        - title must be 2 to 5 words.
        - A good title is specific, emotionally grounded, and reflects the reflection's main moment or shift.
        - Avoid generic titles like "My Day", "Journal Entry", or "Reflection".
        - Use plain language, no emojis, no punctuation-heavy styling.
        - Select 1 to 4 moodTags from this exact allowed list: \(allowedTags.joined(separator: ", ")).
        - moodTags must be lowercase canonical tags from the list, no new tags.
        - If uncertain, choose conservative output with moodTags ["reflective"].

        Never return prose, markdown, context fields, warning fields, or extra keys.
        """
    }

#if canImport(LeapSDK)
    private func generateStructured(systemPrompt: String, message: ChatMessage) async throws -> LiquidStructuredOutput {
        await waitForGenerationTurn()
        defer {
            finishGenerationTurn()
        }
        let runner = try await getRunner()
        let conversation = runner.createConversation(systemPrompt: systemPrompt)
        activeGenerationCount += 1
        defer {
            activeGenerationCount = max(0, activeGenerationCount - 1)
        }

        debugLog("generateStructured start model=\(runner.modelId) descriptor=\(activeModelDescriptor ?? "unknown") parts=\(message.content.count)")

        var options = GenerationOptions()
        options.temperature = 0.5
        options.sequenceLength = configuredSequenceLength()
        options.maxOutputTokens = configuredMaxOutputTokens()
        debugLog(
            "generateStructured limits sequenceLength=\(options.sequenceLength ?? 0) maxOutputTokens=\(options.maxOutputTokens ?? 0)"
        )
        try options.setResponseFormat(type: LiquidInsightPayload.self)

        var jsonText = ""
        var tokensPerSecond: Double?
        var chunkCount = 0

        do {
            for try await response in conversation.generateResponse(message: message, generationOptions: options) {
                switch response {
                case .chunk(let token):
                    chunkCount += 1
                    if chunkCount % 64 == 0 {
                        debugLog("generateStructured chunks=\(chunkCount)")
                    }
                    jsonText += token
                case .complete(let completion):
                    if jsonText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        let fragments = completion.message.content.compactMap { part -> String? in
                            if case .text(let value) = part {
                                return value
                            }
                            return nil
                        }
                        jsonText = fragments.joined()
                    }
                    if let stats = completion.stats {
                        tokensPerSecond = Double(stats.tokenPerSecond)
                        debugLog(
                            "generateStructured complete prompt_tokens=\(stats.promptTokens) completion_tokens=\(stats.completionTokens) total_tokens=\(stats.totalTokens) tps=\(stats.tokenPerSecond)"
                        )
                    } else {
                        debugLog("generateStructured complete without stats")
                    }
                default:
                    break
                }
            }
        } catch {
            errorLog("generateStructured stream failed error=\(detailedErrorText(error))")
            await maybeUnloadModelRunnerIfNeeded(trigger: "stream_error")
            throw error
        }

        guard !jsonText.isEmpty else {
            errorLog("generateStructured failed: empty model output")
            await maybeUnloadModelRunnerIfNeeded(trigger: "empty_output")
            throw LiquidInsightsError.modelOutputMissing
        }

        do {
            let decoded = try decodeLiquidInsightPayload(from: jsonText)
            debugLog(
                "generateStructured decoded title=\(decoded.title) moodTags=\(decoded.moodTags)"
            )
            let output = LiquidStructuredOutput(
                title: decoded.title,
                moodTags: decoded.moodTags,
                modelID: activeModelDescriptor,
                tokensPerSecond: tokensPerSecond
            )
            await maybeUnloadModelRunnerIfNeeded(trigger: "generation_complete")
            return output
        } catch {
            errorLog(
                "generateStructured decode failed error=\(detailedErrorText(error)) chars=\(jsonText.count) raw_preview=\(compactLogPreview(for: jsonText))"
            )
            await maybeUnloadModelRunnerIfNeeded(trigger: "decode_error")
            throw error
        }
    }

    private func maybeUnloadModelRunnerIfNeeded(trigger: String) async {
        guard shouldUnloadAfterEachGeneration() else { return }
        guard activeGenerationCount <= 1 else {
            debugLog("skip unload trigger=\(trigger) activeGenerationCount=\(activeGenerationCount)")
            return
        }
        guard let runner = modelRunner else { return }

        let descriptor = activeModelDescriptor ?? "unknown"
        modelRunner = nil
        activeModelDescriptor = nil
        debugLog("unloading model runner trigger=\(trigger) descriptor=\(descriptor)")
        await runner.unload()
        debugLog("model runner unload complete descriptor=\(descriptor)")
    }

    private func decodeLiquidInsightPayload(from rawText: String) throws -> LiquidInsightPayload {
        func decodeFromData(_ data: Data) throws -> LiquidInsightPayload {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            if let strict = try? decoder.decode(LiquidInsightPayload.self, from: data) {
                return strict
            }

            let object = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = object as? [String: Any] else {
                throw LiquidInsightsError.modelOutputMissing
            }
            return liquidInsightPayload(from: dictionary)
        }

        var firstError: Error?
        if let data = rawText.data(using: .utf8) {
            do {
                return try decodeFromData(data)
            } catch {
                firstError = error
            }
        }

        if let extracted = extractFirstJSONObject(from: rawText),
           let data = extracted.data(using: .utf8) {
            return try decodeFromData(data)
        }

        throw firstError ?? LiquidInsightsError.modelOutputMissing
    }

    private func liquidInsightPayload(from dictionary: [String: Any]) -> LiquidInsightPayload {
        let title = stringValue(
            from: dictionary,
            keys: ["title", "entryTitle", "entry_title", "headline"]
        ) ?? "Daily Reflection"
        let moodTags = stringArray(
            from: dictionary,
            keys: ["moodTags", "mood_tags", "tags", "mood"]
        )

        return LiquidInsightPayload(
            title: title,
            moodTags: moodTags
        )
    }

    private func stringValue(from dictionary: [String: Any], keys: [String]) -> String? {
        for key in keys {
            guard let raw = dictionary[key] else {
                continue
            }
            if let text = raw as? String {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
        }
        return nil
    }

    private func stringArray(from dictionary: [String: Any], keys: [String]) -> [String] {
        for key in keys {
            guard let raw = dictionary[key] else {
                continue
            }
            if let values = raw as? [String] {
                return normalizedStrings(values)
            }
            if let values = raw as? [Any] {
                let mapped = values.compactMap { value -> String? in
                    if let text = value as? String {
                        return text
                    }
                    if let number = value as? NSNumber {
                        return number.stringValue
                    }
                    return nil
                }
                return normalizedStrings(mapped)
            }
            if let single = raw as? String {
                return normalizedStrings([single])
            }
        }
        return []
    }

    private func normalizedStrings(_ values: [String]) -> [String] {
        var output: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                continue
            }
            if output.contains(trimmed) {
                continue
            }
            output.append(trimmed)
        }
        return output
    }

    private func extractFirstJSONObject(from text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else {
            return nil
        }

        var depth = 0
        var inString = false
        var escapeNext = false

        var index = start
        while index < text.endIndex {
            let char = text[index]

            if inString {
                if escapeNext {
                    escapeNext = false
                } else if char == "\\" {
                    escapeNext = true
                } else if char == "\"" {
                    inString = false
                }
            } else {
                if char == "\"" {
                    inString = true
                } else if char == "{" {
                    depth += 1
                } else if char == "}" {
                    depth -= 1
                    if depth == 0 {
                        return String(text[start ... index])
                    }
                }
            }

            index = text.index(after: index)
        }

        return nil
    }

    private func compactLogPreview(for text: String, maxCharacters: Int = 400) -> String {
        let compact = text
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if compact.count <= maxCharacters {
            return compact
        }
        return String(compact.prefix(maxCharacters)) + "..."
    }

    private func getRunner(
        progressHandler: ProgressHandler? = nil,
        forceReload: Bool = false,
        preferManagedRemoteDownload: Bool = false
    ) async throws -> ModelRunner {
        if forceReload {
            debugLog("getRunner forcing reload and clearing cached runner")
            modelRunner = nil
            activeModelDescriptor = nil
        } else if let modelRunner {
            debugLog("getRunner returning cached runner descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(modelRunner.modelId)")
            return modelRunner
        }

        if isLoadingRunner {
            debugLog("getRunner waiting for in-flight model load forceReload=\(forceReload)")
            let runner = try await waitForInFlightRunnerLoad()
            if !forceReload {
                debugLog("getRunner in-flight model load resolved descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
                return runner
            }
            debugLog("getRunner forceReload requested after in-flight load; continuing with reload")
            modelRunner = nil
            activeModelDescriptor = nil
        }

        if !forceReload, let modelRunner {
            debugLog("getRunner returning cached runner after wait descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(modelRunner.modelId)")
            return modelRunner
        }

        isLoadingRunner = true
        defer {
            isLoadingRunner = false
        }

        do {
            let runner = try await loadRunnerForConfiguredModel(
                progressHandler: progressHandler,
                forceReload: forceReload,
                preferManagedRemoteDownload: preferManagedRemoteDownload
            )
            modelRunner = runner
            finishRunnerLoad(with: .success(runner))
            return runner
        } catch {
            finishRunnerLoad(with: .failure(error))
            throw error
        }
    }

    private func loadRunnerForConfiguredModel(
        progressHandler: ProgressHandler? = nil,
        forceReload: Bool = false,
        preferManagedRemoteDownload: Bool = false
    ) async throws -> ModelRunner {
        let defaults = UserDefaults.standard
        let modelName = defaultModel
        let quantization = defaultQuantization
        debugLog("getRunner start model=\(modelName) quantization=\(quantization)")

        if let localPath = defaults.string(forKey: DefaultsKeys.localBundlePath), !localPath.isEmpty {
            let url = URL(fileURLWithPath: localPath)
            debugLog("getRunner using localBundlePath url=\(url.path) file=\(fileDescription(at: url.path))")
            let runner = try loadRunnerFromLocalModel(url: url)
            activeModelDescriptor = url.lastPathComponent
            debugLog("getRunner loaded from localBundlePath descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
            return runner
        }

        let bundledURL = Bundle.main.url(forResource: modelName, withExtension: "bundle", subdirectory: "ModelAssets")
            ?? Bundle.main.url(forResource: modelName, withExtension: "bundle")
        if let bundledURL {
            debugLog("getRunner using bundled .bundle url=\(bundledURL.path) file=\(fileDescription(at: bundledURL.path))")
            let runner = try loadRunnerFromLocalModel(url: bundledURL)
            activeModelDescriptor = "\(modelName).bundle"
            debugLog("getRunner loaded from bundled .bundle descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
            return runner
        }

        if let ggufURL = bundledGGUFURL(modelName: modelName, quantization: quantization) {
            debugLog("getRunner using bundled .gguf url=\(ggufURL.path) file=\(fileDescription(at: ggufURL.path))")
            let runner = try loadRunnerFromLocalModel(url: ggufURL)
            activeModelDescriptor = ggufURL.lastPathComponent
            debugLog("getRunner loaded from bundled .gguf descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
            return runner
        }

        if customModelSourcesEnabled() {
            if let customModelFileURL = configuredURL(forKey: DefaultsKeys.modelFileURL)
                ?? defaultHostedModelFileURL(modelName: modelName, quantization: quantization) {
                debugLog("getRunner attempting custom modelFileURL=\(customModelFileURL.absoluteString)")
                do {
                    let runner = try await loadRunnerFromCustomModelFileURL(
                        modelFileURL: customModelFileURL,
                        progressHandler: progressHandler,
                        forceRedownload: forceReload
                    )
                    debugLog("getRunner custom modelFileURL load success descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
                    return runner
                } catch {
                    errorLog("getRunner custom modelFileURL failed error=\(detailedErrorText(error))")
                }
            }

            if let customManifestURL = configuredURL(forKey: DefaultsKeys.modelManifestURL) {
                debugLog("getRunner attempting custom modelManifestURL=\(customManifestURL.absoluteString)")
                do {
                    let runner = try await loadRunnerFromCustomManifestURL(
                        manifestURL: customManifestURL,
                        progressHandler: progressHandler,
                        forceRedownload: forceReload
                    )
                    debugLog("getRunner custom manifest load success descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
                    return runner
                } catch {
                    errorLog("getRunner custom manifest load failed error=\(detailedErrorText(error))")
                }
            }
        } else {
            debugLog("getRunner custom model sources disabled; using bundled assets or Leap remote defaults")
        }

        if preferManagedRemoteDownload {
            debugLog("getRunner no local model found, using cancellable ModelDownloader path")
#if canImport(LeapModelDownloader)
            activeRemoteDownloadTarget = (model: modelName, quantization: quantization)
            defer {
                if activeRemoteDownloadTarget?.model == modelName,
                   activeRemoteDownloadTarget?.quantization == quantization {
                    activeRemoteDownloadTarget = nil
                }
            }

            if forceReload {
                do {
                    try removeRemoteModelCache(modelName: modelName, quantization: quantization)
                    debugLog("managed remote path forceRedownload cleared cache model=\(modelName) q=\(quantization)")
                } catch {
                    debugLog("managed remote path forceRedownload remove cache error=\(detailedErrorText(error))")
                }
            }

            let runner = try await downloadAndLoadWithModelDownloader(
                modelName: modelName,
                quantization: quantization,
                progressHandler: progressHandler
            )
            debugLog("getRunner managed remote load success descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
            return runner
#else
            debugLog("ModelDownloader unavailable in this build, falling back to Leap remote load")
#endif
        }

        debugLog("getRunner no local model found, using remote load")
        let runner = try await loadRunnerFromRemote(
            modelName: modelName,
            quantization: quantization,
            progressHandler: progressHandler,
            forceRedownload: forceReload
        )
        debugLog("getRunner remote load success descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
        return runner
    }

    func preload() async throws -> String {
        debugLog("preload requested")
        _ = try await getRunner()
        debugLog("preload completed descriptor=\(activeModelDescriptor ?? "liquid_model_loaded")")
        return activeModelDescriptor ?? "liquid_model_loaded"
    }

    func prepareModel(
        forceRedownload: Bool = false,
        progressHandler: ProgressHandler? = nil
    ) async throws -> String {
        debugLog("prepareModel requested forceRedownload=\(forceRedownload)")
        progressHandler?(0)
        _ = try await getRunner(
            progressHandler: progressHandler,
            forceReload: forceRedownload,
            preferManagedRemoteDownload: true
        )
        progressHandler?(1)
        let descriptor = activeModelDescriptor ?? "liquid_model_loaded"
        debugLog("prepareModel completed descriptor=\(descriptor)")
        return descriptor
    }

    func cancelActiveDownloads() async {
        debugLog("cancelActiveDownloads begin")

        if let session = activeCustomDownloadSession {
            debugLog("cancelActiveDownloads canceling custom URLSession download")
            session.invalidateAndCancel()
            activeCustomDownloadSession = nil
        }

#if canImport(LeapModelDownloader)
        if let target = activeRemoteDownloadTarget {
            let downloadableModel = await LeapDownloadableModel.resolve(
                modelSlug: target.model,
                quantizationSlug: target.quantization
            )

            if let activeModelDownloader {
                debugLog("cancelActiveDownloads stopping active ModelDownloader instance for model=\(target.model) quantization=\(target.quantization)")
                if let downloadableModel {
                    activeModelDownloader.requestStopDownload(downloadableModel)
                }
                activeModelDownloader.requestStopService()
                self.activeModelDownloader = nil
            }

            let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
            if let downloadableModel {
                downloader.requestStopDownload(downloadableModel)
            }
            downloader.requestStopService()

            let status = downloader.queryStatus(target.model, quantization: target.quantization)
            debugLog(
                "cancelActiveDownloads model status before stop model=\(target.model) quantization=\(target.quantization) status=\(downloadStatusDescription(status))"
            )

            do {
                try downloader.removeModel(target.model, quantization: target.quantization)
                debugLog("cancelActiveDownloads removed cached/partial model model=\(target.model) quantization=\(target.quantization)")
            } catch {
                debugLog(
                    "cancelActiveDownloads removeModel skipped model=\(target.model) quantization=\(target.quantization) error=\(detailedErrorText(error))"
                )
            }
        } else {
            if let activeModelDownloader {
                debugLog("cancelActiveDownloads stopping active ModelDownloader instance")
                activeModelDownloader.requestStopService()
                self.activeModelDownloader = nil
            }

            let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
            downloader.requestStopService()
        }
#endif

        debugLog("cancelActiveDownloads end")
    }

    private func bundledGGUFURL(modelName: String, quantization: String) -> URL? {
        let directCandidates = [
            "\(modelName)-\(quantization)",
            "\(modelName)_\(quantization)",
            "\(modelName).\(quantization)",
            modelName,
        ]

        for candidate in directCandidates {
            if let url = Bundle.main.url(forResource: candidate, withExtension: "gguf", subdirectory: "ModelAssets") {
                return url
            }
        }

        for candidate in directCandidates {
            if let url = Bundle.main.url(forResource: candidate, withExtension: "gguf") {
                return url
            }
        }

        var resources: [URL] = []
        if let root = Bundle.main.urls(forResourcesWithExtension: "gguf", subdirectory: nil) {
            resources.append(contentsOf: root)
        }
        if let modelAssets = Bundle.main.urls(forResourcesWithExtension: "gguf", subdirectory: "ModelAssets") {
            resources.append(contentsOf: modelAssets)
        }

        guard !resources.isEmpty else {
            return nil
        }

        let normalizedModel = normalizedResourceKey(modelName)
        let normalizedQuantization = normalizedResourceKey(quantization)
        let ranked = resources.compactMap { url -> (url: URL, score: Int)? in
            let stem = normalizedResourceKey(url.deletingPathExtension().lastPathComponent)
            var score = 0
            if stem.contains(normalizedModel) {
                score += 2
            }
            if !normalizedQuantization.isEmpty && stem.contains(normalizedQuantization) {
                score += 1
            }
            return score > 0 ? (url, score) : nil
        }.sorted {
            if $0.score != $1.score {
                return $0.score > $1.score
            }
            return $0.url.lastPathComponent < $1.url.lastPathComponent
        }

        let selected = ranked.first?.url
        if let selected {
            debugLog("bundledGGUFURL selected \(selected.lastPathComponent)")
        }
        return selected
    }

    private func normalizedResourceKey(_ value: String) -> String {
        value
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined()
    }

    private func configuredURL(forKey key: String) -> URL? {
        guard let rawValue = UserDefaults.standard.string(forKey: key)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !rawValue.isEmpty else {
            return nil
        }

        if rawValue.hasPrefix("/") {
            return URL(fileURLWithPath: rawValue)
        }

        if let url = URL(string: rawValue), url.scheme != nil {
            return url
        }

        return nil
    }

    private func defaultHostedModelFileURL(modelName: String, quantization: String) -> URL? {
        guard let baseURL = URL(string: defaultHostedModelBaseURL) else {
            return nil
        }

        let filename = "\(modelName)-\(quantization).gguf"
        return baseURL.appendingPathComponent(filename)
    }

    private func loadRunnerFromCustomModelFileURL(
        modelFileURL: URL,
        progressHandler: ProgressHandler?,
        forceRedownload: Bool
    ) async throws -> ModelRunner {
        if modelFileURL.isFileURL {
            let path = modelFileURL.path
            debugLog("custom modelFileURL uses local file path=\(path) file=\(fileDescription(at: path))")
            guard FileManager.default.fileExists(atPath: path) else {
                throw NSError(
                    domain: "LiquidInsightsRuntime",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Custom model file not found at \(path)"]
                )
            }
            progressHandler?(1)
            let runner = try loadRunnerFromLocalModel(url: modelFileURL)
            activeModelDescriptor = modelFileURL.lastPathComponent
            return runner
        }

        let cachedURL = try cachedCustomModelURL(for: modelFileURL)
        let fileManager = FileManager.default

        if forceRedownload, fileManager.fileExists(atPath: cachedURL.path) {
            debugLog("forceRedownload removing cached custom model path=\(cachedURL.path)")
            try? fileManager.removeItem(at: cachedURL)
        }

        if !fileManager.fileExists(atPath: cachedURL.path) {
            var lastError: Error?
            for attempt in 1...remoteLoadMaxAttempts {
                do {
                    debugLog("custom model download attempt=\(attempt) url=\(modelFileURL.absoluteString)")
                    progressHandler?(0.02)
                    try await downloadCustomModelFile(
                        from: modelFileURL,
                        to: cachedURL,
                        progressHandler: progressHandler,
                        progressRange: 0.02...0.95
                    )
                    break
                } catch {
                    lastError = error
                    errorLog("custom model download failed attempt=\(attempt) error=\(detailedErrorText(error))")
                    guard attempt < remoteLoadMaxAttempts, shouldRetryRemoteLoad(after: error) else {
                        throw error
                    }
                    let delay = retryDelayNanoseconds(forAttempt: attempt)
                    try? await Task.sleep(nanoseconds: delay)
                }
            }

            if let lastError, !fileManager.fileExists(atPath: cachedURL.path) {
                throw lastError
            }
        } else {
            debugLog("using cached custom model file path=\(cachedURL.path) file=\(fileDescription(at: cachedURL.path))")
        }

        progressHandler?(0.98)
        let runner = try loadRunnerFromLocalModel(url: cachedURL)
        activeModelDescriptor = cachedURL.lastPathComponent
        progressHandler?(1)
        return runner
    }

    private func loadRunnerFromCustomManifestURL(
        manifestURL: URL,
        progressHandler: ProgressHandler?,
        forceRedownload: Bool
    ) async throws -> ModelRunner {
#if canImport(LeapModelDownloader)
        if forceRedownload {
            do {
                let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
                try downloader.removeModel(fromManifestURL: manifestURL)
                debugLog("forceRedownload removed cached manifest model manifestURL=\(manifestURL.absoluteString)")
            } catch {
                debugLog("forceRedownload manifest cache remove error=\(detailedErrorText(error))")
            }
        }
#endif

        let verbose = isVerboseLoggingEnabled()
        let logger = self.logger
        let runner = try await Leap.load(
            manifestURL: manifestURL,
            options: nil,
            downloadProgressHandler: { progress, speed in
                progressHandler?(max(0, min(1, progress)))
                guard verbose else { return }
                let percentage = Int((progress * 100).rounded())
                let mbps = Double(speed) / 1_048_576
                logger.debug(
                    "custom manifest progress url=\(manifestURL.absoluteString, privacy: .public) progress=\(percentage, privacy: .public)% speedMBps=\(mbps, privacy: .public)"
                )
            }
        )
        progressHandler?(1)
        activeModelDescriptor = manifestURL.lastPathComponent.isEmpty
            ? "custom-manifest"
            : manifestURL.lastPathComponent
        return runner
    }

    private struct CustomModelDownloadResult {
        let temporaryURL: URL
        let response: URLResponse
    }

    private final class CustomModelDownloadDelegate: NSObject, URLSessionDownloadDelegate {
        private let initialBytes: Int64
        private let progressRange: ClosedRange<Double>
        private let progressHandler: ProgressHandler?

        private var continuation: CheckedContinuation<CustomModelDownloadResult, Error>?
        private var downloadedURL: URL?
        private var response: URLResponse?
        private var hasResolved = false

        init(
            initialBytes: Int64,
            progressRange: ClosedRange<Double>,
            progressHandler: ProgressHandler?
        ) {
            self.initialBytes = max(0, initialBytes)
            self.progressRange = progressRange
            self.progressHandler = progressHandler
        }

        func start(session: URLSession, request: URLRequest) async throws -> CustomModelDownloadResult {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                let task = session.downloadTask(with: request)
                task.resume()
            }
        }

        private func resolve(_ result: Result<CustomModelDownloadResult, Error>) {
            guard !hasResolved else { return }
            hasResolved = true
            continuation?.resume(with: result)
            continuation = nil
        }

        private func reportProgress(totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
            guard totalBytesExpectedToWrite > 0 else {
                return
            }

            let expectedOverall = initialBytes + totalBytesExpectedToWrite
            guard expectedOverall > 0 else {
                return
            }

            let writtenOverall = max(0, min(expectedOverall, initialBytes + totalBytesWritten))
            let fraction = max(0, min(1, Double(writtenOverall) / Double(expectedOverall)))
            let mapped = progressRange.lowerBound + ((progressRange.upperBound - progressRange.lowerBound) * fraction)
            progressHandler?(mapped)
        }

        func urlSession(
            _: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData _: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            var expected = totalBytesExpectedToWrite
            if expected <= 0, let response = downloadTask.response {
                expected = response.expectedContentLength
            }
            reportProgress(
                totalBytesWritten: totalBytesWritten,
                totalBytesExpectedToWrite: expected
            )
        }

        func urlSession(
            _: URLSession,
            downloadTask _: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            downloadedURL = location
        }

        func urlSession(
            _: URLSession,
            task: URLSessionTask,
            didCompleteWithError error: (any Error)?
        ) {
            response = task.response
            if let error {
                resolve(.failure(error))
                return
            }

            guard let downloadedURL else {
                resolve(
                    .failure(
                        NSError(
                            domain: "LiquidInsightsRuntime",
                            code: 4,
                            userInfo: [NSLocalizedDescriptionKey: "Model download did not produce a file."]
                        )
                    )
                )
                return
            }

            guard let response else {
                resolve(
                    .failure(
                        NSError(
                            domain: "LiquidInsightsRuntime",
                            code: 5,
                            userInfo: [NSLocalizedDescriptionKey: "Model download finished without a response."]
                        )
                    )
                )
                return
            }

            resolve(
                .success(
                    CustomModelDownloadResult(
                        temporaryURL: downloadedURL,
                        response: response
                    )
                )
            )
        }
    }

    private func downloadCustomModelFile(
        from remoteURL: URL,
        to destinationURL: URL,
        progressHandler: ProgressHandler?,
        progressRange: ClosedRange<Double> = 0...0.95
    ) async throws {
        let configuration = customModelDownloadSessionConfiguration()
        let delegate = CustomModelDownloadDelegate(
            initialBytes: 0,
            progressRange: progressRange,
            progressHandler: progressHandler
        )
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        activeCustomDownloadSession = session
        defer {
            if activeCustomDownloadSession === session {
                activeCustomDownloadSession = nil
            }
            session.invalidateAndCancel()
        }
        progressHandler?(progressRange.lowerBound)

        var request = URLRequest(url: remoteURL)
        request.timeoutInterval = configuration.timeoutIntervalForRequest

        let result = try await delegate.start(session: session, request: request)
        let temporaryURL = result.temporaryURL
        let response = result.response

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw NSError(
                domain: NSURLErrorDomain,
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "Custom model download failed with status \(http.statusCode)"]
            )
        }

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destinationURL.path) {
            try? fileManager.removeItem(at: destinationURL)
        }

        let parent = destinationURL.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: parent.path) {
            try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        }

        do {
            try fileManager.moveItem(at: temporaryURL, to: destinationURL)
        } catch {
            try fileManager.copyItem(at: temporaryURL, to: destinationURL)
        }

        let fileSize = (try? fileManager.attributesOfItem(atPath: destinationURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        guard fileSize > 0 else {
            throw NSError(
                domain: "LiquidInsightsRuntime",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Downloaded custom model is empty at \(destinationURL.path)"]
            )
        }

        progressHandler?(progressRange.upperBound)
        debugLog("custom model downloaded path=\(destinationURL.path) sizeBytes=\(fileSize)")
    }

    private func customModelDownloadSessionConfiguration() -> URLSessionConfiguration {
#if canImport(LeapModelDownloader)
        let configuration = URLSessionConfiguration.leapDefault
#else
        let configuration = URLSessionConfiguration.default
#endif
        configuration.waitsForConnectivity = true
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 60 * 60
        return configuration
    }

    private func cachedCustomModelURL(for remoteURL: URL) throws -> URL {
        let cacheDirectory = try customModelCacheDirectory()
        let digest = SHA256.hash(data: Data(remoteURL.absoluteString.utf8))
        let digestPrefix = digest.map { String(format: "%02x", $0) }.joined().prefix(12)
        let originalName = remoteURL.lastPathComponent.isEmpty ? "model.gguf" : remoteURL.lastPathComponent
        let safeName = sanitizeFilename(originalName)
        return cacheDirectory.appendingPathComponent("\(digestPrefix)-\(safeName)")
    }

    private func customModelCacheDirectory() throws -> URL {
        let fileManager = FileManager.default
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = appSupport.appendingPathComponent("LiquidModelCache", isDirectory: true)
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }

    private func sanitizeFilename(_ value: String) -> String {
        let cleaned = value.replacingOccurrences(
            of: "[^A-Za-z0-9._-]",
            with: "_",
            options: .regularExpression
        )
        return cleaned.isEmpty ? "model.gguf" : cleaned
    }

    private func loadRunnerFromLocalModel(url: URL) throws -> ModelRunner {
        debugLog("loadRunnerFromLocalModel path=\(url.path) file=\(fileDescription(at: url.path))")
        let options = LiquidInferenceEngineOptions(bundlePath: url.path)
        let runner = try Leap.load(options: options)
        debugLog("loadRunnerFromLocalModel success modelId=\(runner.modelId)")
        return runner
    }

    private func loadRunnerFromRemote(
        modelName: String,
        quantization: String,
        progressHandler: ProgressHandler?,
        forceRedownload: Bool
    ) async throws -> ModelRunner {
        debugLog("loadRunnerFromRemote start model=\(modelName) quantization=\(quantization)")
        do {
            return try await loadRunnerFromRemoteForSingleQuantization(
                modelName: modelName,
                quantization: quantization,
                progressHandler: progressHandler,
                forceRedownload: forceRedownload
            )
        } catch {
            var lastError: Error = error
            errorLog("primary quantization load failed q=\(quantization) error=\(detailedErrorText(error))")

#if canImport(LeapModelDownloader)
            guard shouldTryAlternateQuantizations(after: error) else {
                throw error
            }

            for alternate in alternateQuantizations(from: quantization) {
                debugLog("trying alternate quantization \(alternate)")
                do {
                    let runner = try await loadRunnerFromRemoteForSingleQuantization(
                        modelName: modelName,
                        quantization: alternate,
                        progressHandler: progressHandler,
                        forceRedownload: forceRedownload
                    )
                    debugLog("alternate quantization succeeded q=\(alternate)")
                    return runner
                } catch {
                    errorLog("alternate quantization failed q=\(alternate) error=\(detailedErrorText(error))")
                    lastError = error
                }
            }
#endif

            errorLog("all remote quantization attempts failed model=\(modelName)")
            throw lastError
        }
    }

    private func loadRunnerFromRemoteForSingleQuantization(
        modelName: String,
        quantization: String,
        progressHandler: ProgressHandler?,
        forceRedownload: Bool
    ) async throws -> ModelRunner {
        var lastError: Error?
        activeRemoteDownloadTarget = (model: modelName, quantization: quantization)
        defer {
            if activeRemoteDownloadTarget?.model == modelName,
               activeRemoteDownloadTarget?.quantization == quantization {
                activeRemoteDownloadTarget = nil
            }
        }

#if canImport(LeapModelDownloader)
        if forceRedownload {
            do {
                try removeRemoteModelCache(modelName: modelName, quantization: quantization)
                debugLog("forceRedownload cleared cache model=\(modelName) q=\(quantization)")
            } catch {
                debugLog("forceRedownload remove cache error=\(detailedErrorText(error))")
            }
        }
#endif

        for attempt in 1...remoteLoadMaxAttempts {
            debugLog("remote load attempt=\(attempt) model=\(modelName) quantization=\(quantization)")
            do {
                let verbose = isVerboseLoggingEnabled()
                let logger = self.logger
                let runner = try await Leap.load(
                    model: modelName,
                    quantization: quantization,
                    options: nil,
                    downloadProgressHandler: { progress, speed in
                        progressHandler?(max(0, min(1, progress)))
                        guard verbose else { return }
                        let percentage = Int((progress * 100).rounded())
                        let mbps = Double(speed) / 1_048_576
                        logger.debug(
                            "remote download progress model=\(modelName, privacy: .public) q=\(quantization, privacy: .public) attempt=\(attempt, privacy: .public) progress=\(percentage, privacy: .public)% speedMBps=\(mbps, privacy: .public)"
                        )
                    }
                )
                progressHandler?(1)
                activeModelDescriptor = "\(modelName)@\(quantization)"
                debugLog("remote load success attempt=\(attempt) descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
                return runner
            } catch {
                lastError = error
                errorLog("remote load failed attempt=\(attempt) model=\(modelName) q=\(quantization) error=\(detailedErrorText(error))")

#if canImport(LeapModelDownloader)
                if shouldTreatAsCorruptOrMissingModel(after: error) {
                    debugLog("detected corrupt/missing model artifacts, forcing remove + redownload")
                    do {
                        return try await redownloadAndLoadAfterCorruption(
                            modelName: modelName,
                            quantization: quantization,
                            progressHandler: progressHandler
                        )
                    } catch {
                        errorLog("redownload after corruption failed error=\(detailedErrorText(error))")
                        lastError = error
                    }
                }

                if shouldUseDownloaderFallback(after: error) {
                    debugLog("detected timeout path, switching to explicit ModelDownloader fallback")
                    do {
                        return try await downloadAndLoadWithModelDownloader(
                            modelName: modelName,
                            quantization: quantization,
                            progressHandler: progressHandler
                        )
                    } catch {
                        errorLog("ModelDownloader fallback failed error=\(detailedErrorText(error))")
                        lastError = error
                    }
                }
#endif

                guard attempt < remoteLoadMaxAttempts, shouldRetryRemoteLoad(after: error) else {
                    debugLog("remote load will not retry attempt=\(attempt)")
                    break
                }

                let delay = retryDelayNanoseconds(forAttempt: attempt)
                debugLog("retrying after \(Double(delay) / 1_000_000_000)s")
                try? await Task.sleep(nanoseconds: delay)
            }
        }

        errorLog("remote load exhausted attempts model=\(modelName) q=\(quantization)")
        throw lastError ?? LiquidInsightsError.unavailable
    }

    private func shouldRetryRemoteLoad(after error: Error) -> Bool {
        if error is CancellationError {
            return false
        }

        let retryableCodes: Set<Int> = [
            NSURLErrorTimedOut,
            NSURLErrorNetworkConnectionLost,
            NSURLErrorCannotConnectToHost,
            NSURLErrorCannotFindHost,
            NSURLErrorDNSLookupFailed,
            NSURLErrorNotConnectedToInternet,
            NSURLErrorDataNotAllowed,
            NSURLErrorCallIsActive,
            NSURLErrorInternationalRoamingOff,
        ]

        return errorChain(error).contains { nsError in
            nsError.domain == NSURLErrorDomain && retryableCodes.contains(nsError.code)
        }
    }

    private func retryDelayNanoseconds(forAttempt attempt: Int) -> UInt64 {
        UInt64(max(1, attempt)) * 1_500_000_000
    }

    private func errorChain(_ error: Error) -> [NSError] {
        var chain: [NSError] = []
        var next: NSError? = error as NSError
        var seen = Set<ObjectIdentifier>()

        while let current = next {
            let id = ObjectIdentifier(current)
            if seen.contains(id) {
                break
            }
            seen.insert(id)
            chain.append(current)
            next = current.userInfo[NSUnderlyingErrorKey] as? NSError
        }

        return chain
    }

    private func detailedErrorText(_ error: Error) -> String {
        var parts: [String] = [String(describing: error)]
        for nsError in errorChain(error) {
            parts.append(nsError.localizedDescription)
            if let reason = nsError.localizedFailureReason {
                parts.append(reason)
            }
            if let description = nsError.userInfo[NSLocalizedDescriptionKey] as? String {
                parts.append(description)
            }
        }

        return parts
            .joined(separator: " | ")
            .lowercased()
    }

#if canImport(LeapModelDownloader)
    private func shouldTryAlternateQuantizations(after error: Error) -> Bool {
        let text = detailedErrorText(error)
        let checks = [
            "error loading model: vector",
            "failed to initialize model and context",
            "failed to load model",
            "llama_model_load",
        ]
        return checks.contains { text.contains($0) }
    }

    private func alternateQuantizations(from quantization: String) -> [String] {
        let ordered = ["Q4_K_M", "Q4_0", "Q8_0"]
        return ordered.filter { $0.caseInsensitiveCompare(quantization) != .orderedSame }
    }

    private func shouldTreatAsCorruptOrMissingModel(after error: Error) -> Bool {
        let text = detailedErrorText(error)
        let checks = [
            "fopen failed for data file",
            "failed to initialize model and context",
            "no such file or directory",
            "error loading model",
            "failed to load model",
        ]

        return checks.contains { text.contains($0) }
    }

    private func redownloadAndLoadAfterCorruption(
        modelName: String,
        quantization: String,
        progressHandler: ProgressHandler?
    ) async throws -> ModelRunner {
        let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
        do {
            try downloader.removeModel(modelName, quantization: quantization)
            debugLog("removed cached model before redownload model=\(modelName) q=\(quantization)")
        } catch {
            debugLog("removeModel before redownload returned error=\(detailedErrorText(error))")
        }
        return try await downloadAndLoadWithModelDownloader(
            modelName: modelName,
            quantization: quantization,
            progressHandler: progressHandler
        )
    }

    private func removeRemoteModelCache(modelName: String, quantization: String) throws {
        let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
        try downloader.removeModel(modelName, quantization: quantization)
    }

    private func shouldUseDownloaderFallback(after error: Error) -> Bool {
        errorChain(error).contains { nsError in
            nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorTimedOut
        }
    }

    private func downloadAndLoadWithModelDownloader(
        modelName: String,
        quantization: String,
        progressHandler: ProgressHandler?
    ) async throws -> ModelRunner {
        debugLog("ModelDownloader start model=\(modelName) quantization=\(quantization)")
        let downloader = ModelDownloader(sessionConfiguration: modelDownloaderSessionConfiguration())
        let downloadableModel = await LeapDownloadableModel.resolve(
            modelSlug: modelName,
            quantizationSlug: quantization
        )
        activeModelDownloader = downloader
        defer {
            if activeModelDownloader === downloader {
                activeModelDownloader = nil
            }
        }
        let verbose = isVerboseLoggingEnabled()
        let logger = self.logger
        let manifest: DownloadedModelManifest
        do {
            manifest = try await withTaskCancellationHandler(
                operation: {
                    try Task.checkCancellation()
                    return try await downloader.downloadModel(
                        modelName,
                        quantization: quantization,
                        downloadProgress: { progress, speed in
                            progressHandler?(max(0, min(1, progress)))
                            guard verbose else { return }
                            let percentage = Int((progress * 100).rounded())
                            let mbps = Double(speed) / 1_048_576
                            logger.debug(
                                "downloader progress model=\(modelName, privacy: .public) q=\(quantization, privacy: .public) progress=\(percentage, privacy: .public)% speedMBps=\(mbps, privacy: .public)"
                            )
                        }
                    )
                },
                onCancel: {
                    if let downloadableModel {
                        downloader.requestStopDownload(downloadableModel)
                    }
                    downloader.requestStopService()
                }
            )
        } catch let error as ModelDownloadError {
            if case .downloadCancelled = error {
                debugLog("ModelDownloader canceled by caller model=\(modelName) quantization=\(quantization)")
                throw CancellationError()
            }
            throw error
        } catch is CancellationError {
            debugLog("ModelDownloader task canceled model=\(modelName) quantization=\(quantization)")
            throw CancellationError()
        }
        debugLog(
            "ModelDownloader manifest modelURL=\(manifest.localModelURL.path) mmProj=\(manifest.localMultimodalProjectorURL?.path ?? "nil") audioDecoder=\(manifest.localAudioDecoderURL?.path ?? "nil") audioTokenizer=\(manifest.localAudioTokenizerURL?.path ?? "nil")"
        )

        try Task.checkCancellation()

        let options = LiquidInferenceEngineOptions(
            bundlePath: manifest.localModelURL.path,
            mmProjPath: manifest.localMultimodalProjectorURL?.path,
            audioDecoderPath: manifest.localAudioDecoderURL?.path,
            chatTemplate: manifest.chatTemplate,
            audioTokenizerPath: manifest.localAudioTokenizerURL?.path
        )

        guard FileManager.default.fileExists(atPath: manifest.localModelURL.path) else {
            errorLog("ModelDownloader missing model file path=\(manifest.localModelURL.path)")
            throw NSError(
                domain: "LiquidInsightsRuntime",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Downloaded model file missing at \(manifest.localModelURL.path)"]
            )
        }

        debugLog("ModelDownloader file check modelFile=\(fileDescription(at: manifest.localModelURL.path))")
        let runner = try Leap.load(options: options)
        progressHandler?(1)
        activeModelDescriptor = manifest.localModelURL.lastPathComponent
        debugLog("ModelDownloader load success descriptor=\(activeModelDescriptor ?? "unknown") modelId=\(runner.modelId)")
        return runner
    }

    private func modelDownloaderSessionConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.leapDefault
        configuration.waitsForConnectivity = true
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 60 * 60
        debugLog(
            "ModelDownloader session config requestTimeout=\(configuration.timeoutIntervalForRequest)s resourceTimeout=\(configuration.timeoutIntervalForResource)s waitsForConnectivity=\(configuration.waitsForConnectivity)"
        )
        return configuration
    }
#endif
#else
    private func generateStructured(systemPrompt _: String, message _: ChatMessage) async throws -> LiquidStructuredOutput {
        throw LiquidInsightsError.unavailable
    }
#endif
}

#if canImport(LeapSDK)
@Generatable("Structured extraction for entry title and mood tags.")
private struct LiquidInsightPayload: Codable {
    @Guide("A concise 2 to 5 word reflection title.")
    let title: String

    @Guide("1 to 4 canonical mood tags from the allowed list.")
    let moodTags: [String]
}
#else
    private struct ChatMessage {
        enum Role {
            case user
        }

        enum Content {
            case text(String)
        }

        let role: Role
        let content: [Content]
}
#endif
