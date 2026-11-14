import Foundation

/// The OpenAI-compatible endpoint (SPEC §4): `GET /v1/models` and `POST /v1/chat/completions`, streaming
/// included. Requests go through the same queue and prompt-cache slots as the app's own chats.
@MainActor
public final class RuntimeAPIEndpoint {
    unowned let engine: ParlorEngine
    public private(set) var server: LoopbackServer?

    init(engine: ParlorEngine) { self.engine = engine }

    public var port: Int { Int(server?.port ?? 0) }
    public var isRunning: Bool { server != nil }

    public func start(port: Int) async throws {
        stop()
        let s = LoopbackServer { [weak self] req in
            guard let self else { return .init(status: 404) }
            return await self.handle(req)
        }
        try await s.start(port: port)
        server = s
    }

    public func stop() { server?.stop(); server = nil }

    // MARK: errors

    nonisolated static func error(_ status: Int, _ message: String, type: String = "invalid_request_error", code: String? = nil) -> LoopbackServer.Response {
        var e: [String: JSON] = ["message": .string(message), "type": .string(type)]
        if let code { e["code"] = .string(code) }
        return .init(status: status, body: OpenAIWire.encode(.object(["error": .object(e)])))
    }

    private static let unsupportedPaths = ["/v1/embeddings", "/v1/images", "/v1/audio", "/v1/completions", "/v1/responses", "/v1/moderations"]

    // MARK: routing

    func handle(_ req: LoopbackServer.Request) async -> LoopbackServer.Response {
        if let o = req.headers["origin"], !o.isEmpty, !(o.contains("://127.0.0.1") || o.contains("://localhost")) { return .init(status: 403) }
        if let h = req.headers["host"], !(h.hasPrefix("127.0.0.1") || h.hasPrefix("localhost") || h.hasPrefix("[::1]")) { return .init(status: 403) }
        let path = req.path.split(separator: "?").first.map(String.init) ?? req.path
        guard path.hasPrefix("/v1/") else { return .init(status: 404) }
        let auth = req.headers["authorization"] ?? ""
        guard auth.lowercased().hasPrefix("bearer "), engine.endpoint.authenticate(String(auth.dropFirst(7)).trimmingCharacters(in: .whitespaces)) != nil else {
            return Self.error(401, "Invalid or missing API key. Send Authorization: Bearer <token>.", code: "invalid_api_key")
        }
        switch (req.method, path) {
        case ("GET", "/v1/models"): return await models()
        case ("POST", "/v1/chat/completions"): return await chat(req.body)
        default:
            if Self.unsupportedPaths.contains(where: { path.hasPrefix($0) }) {
                return Self.error(400, "This endpoint is not supported. Parlor serves /v1/models and /v1/chat/completions only (no embeddings, images, audio or legacy completions).", code: "unsupported")
            }
            return Self.error(404, "Unknown route \(req.method) \(path).", code: "not_found")
        }
    }

    private func models() async -> LoopbackServer.Response {
        let items = await engine.runtimeModelNames().map { name -> JSON in
            .object(["id": .string(name), "object": .string("model"), "created": .int(0),
                     "owned_by": .string(name.split(separator: "/").first.map(String.init) ?? "parlor")])
        }
        return .init(status: 200, body: OpenAIWire.encode(.object(["object": .string("list"), "data": .array(items)])))
    }

    // MARK: chat completions

    private final class Pump: @unchecked Sendable {
        var iterator: AsyncThrowingStream<LLMEvent, Error>.Iterator
        var first: LLMEvent?
        init(_ it: AsyncThrowingStream<LLMEvent, Error>.Iterator, first: LLMEvent?) { iterator = it; self.first = first }
        func next() async throws -> LLMEvent? {
            if let f = first { first = nil; return f }
            return try await iterator.next()
        }
    }

    private func chat(_ body: Data) async -> LoopbackServer.Response {
        guard let o = (try? JSONDecoder().decode(JSON.self, from: body))?.objectValue else {
            return Self.error(400, "The body is not a JSON object.")
        }
        guard let model = o["model"]?.stringValue, !model.isEmpty else { return Self.error(400, "\"model\" is required.", code: "missing_model") }
        guard let messages = OpenAIWire.parseMessages(o["messages"]), !messages.isEmpty else {
            return Self.error(400, "\"messages\" must be a non-empty list of system, user, assistant or tool messages.", code: "bad_messages")
        }
        if let n = o["n"]?.intValue, n > 1 { return Self.error(400, "n > 1 is not supported.", code: "unsupported") }
        if o["logprobs"] == .bool(true) || (o["top_logprobs"]?.intValue ?? 0) > 0 { return Self.error(400, "logprobs are not supported.", code: "unsupported") }
        guard engine.runtimeKnowsModel(model) else {
            return Self.error(404, "The model \"\(model)\" is not installed or not provided by an enabled provider. GET /v1/models lists what is available.", code: "model_not_found")
        }
        let stream = o["stream"] == .bool(true)
        let includeUsage = o["stream_options"]?.objectValue?["include_usage"] == .bool(true)
        var tuning = LLMTuning(settings: engine.settings.llm)
        if case .double(let p)? = o["top_p"] { tuning.settings.topP = p } else if let p = o["top_p"]?.intValue { tuning.settings.topP = Double(p) }
        var temperature = 0.7
        if case .double(let t)? = o["temperature"] { temperature = t } else if let t = o["temperature"]?.intValue { temperature = Double(t) }
        let maxTokens = (o["max_completion_tokens"] ?? o["max_tokens"])?.intValue
        let thinking = o["chat_template_kwargs"]?.objectValue?["enable_thinking"] == .bool(true)
        let request = LLMRequest(messages: messages, tools: OpenAIWire.parseTools(o["tools"]), temperature: temperature,
                                 maxTokens: maxTokens, thinking: thinking, tuning: tuning, priority: .user)
        let events = engine.runtime.generate(model: model, request: request)
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
        let created = Int(Date().timeIntervalSince1970)

        // Pull the first event before answering, so a model that cannot load is a clean 404/503 rather than a broken stream.
        let pump = Pump(events.makeAsyncIterator(), first: nil)
        do { pump.first = try await pump.next() } catch {
            return Self.failure(error)
        }

        if !stream {
            var text = ""
            var calls: [ToolCallRecord] = []
            var stats: LLMStats?
            do {
                while let ev = try await pump.next() {
                    switch ev {
                    case .text(let t): text += t
                    case .toolCall(let c): calls.append(c)
                    case .stats(let s): stats = s
                    }
                }
            } catch { return Self.failure(error) }
            var msg: [String: JSON] = ["role": .string("assistant"), "content": calls.isEmpty || !text.isEmpty ? .string(text) : .null]
            if !calls.isEmpty { msg["tool_calls"] = .array(calls.map(Self.toolCallJSON)) }
            var out: [String: JSON] = [
                "id": .string(id), "object": .string("chat.completion"), "created": .int(created), "model": .string(model),
                "choices": .array([.object(["index": .int(0), "message": .object(msg), "finish_reason": .string(calls.isEmpty ? "stop" : "tool_calls")])])]
            out["usage"] = Self.usage(stats)
            return .init(status: 200, body: OpenAIWire.encode(.object(out)))
        }

        return LoopbackServer.Response(streaming: "text/event-stream") { writer in
            func send(_ delta: [String: JSON], finish: String? = nil, usage: JSON? = nil, noChoices: Bool = false) async -> Bool {
                var o: [String: JSON] = ["id": .string(id), "object": .string("chat.completion.chunk"), "created": .int(created), "model": .string(model)]
                o["choices"] = noChoices ? .array([]) : .array([.object(["index": .int(0), "delta": .object(delta), "finish_reason": finish.map { .string($0) } ?? .null])])
                if let usage { o["usage"] = usage }
                return await writer.write(Data(("data: " + JSON.object(o).text + "\n\n").utf8))
            }
            guard await send(["role": .string("assistant"), "content": .string("")]) else { return }
            var calls = 0
            var stats: LLMStats?
            do {
                while let ev = try await pump.next() {
                    switch ev {
                    case .text(let t): guard await send(["content": .string(t)]) else { return }   // client gone: dropping the pump cancels generation
                    case .toolCall(let c):
                        var j = Self.toolCallJSON(c).objectValue ?? [:]
                        j["index"] = .int(calls); calls += 1
                        guard await send(["tool_calls": .array([.object(j)])]) else { return }
                    case .stats(let s): stats = s
                    }
                }
                guard await send([:], finish: calls > 0 ? "tool_calls" : "stop") else { return }
                if includeUsage { _ = await send([:], usage: Self.usage(stats), noChoices: true) }
            } catch {
                let e = JSON.object(["error": .object(["message": .string(error.localizedDescription), "type": .string("server_error")])])
                _ = await writer.write(Data(("data: " + e.text + "\n\n").utf8))
            }
            _ = await writer.write(Data("data: [DONE]\n\n".utf8))
        }
    }

    nonisolated static func toolCallJSON(_ c: ToolCallRecord) -> JSON {
        .object(["id": .string(c.id), "type": .string("function"),
                 "function": .object(["name": .string(OpenAIWire.externalName(c.name)), "arguments": .string(c.arguments.text)])])
    }

    nonisolated static func usage(_ s: LLMStats?) -> JSON {
        let p = s?.promptTokens ?? 0, c = s?.completionTokens ?? 0
        return .object(["prompt_tokens": .int(p), "completion_tokens": .int(c), "total_tokens": .int(p + c),
                        "prompt_tokens_details": .object(["cached_tokens": .int(s?.cachedTokens ?? 0)])])
    }

    nonisolated static func failure(_ error: Error) -> LoopbackServer.Response {
        if case ParlorError.modelNotInstalled(let m) = error {
            return Self.error(404, "The model \"\(m)\" is not installed.", code: "model_not_found")
        }
        return Self.error(503, error.localizedDescription, type: "server_error", code: "model_unavailable")
    }
}
