import Foundation

/// The routes of the one loopback server (`/v1/*`; `/mcp` lives beside it): `GET /v1/models`, `POST /v1/chat/completions`
/// and `POST /v1/images/generations`. Everything that asks a host to do something goes through here: the app's own
/// chats, harnesses, the image tool and outside clients share the same queues, priorities, memory budget and scopes.
@MainActor
public final class GatewayRoutes {
    unowned let engine: ParlorEngine

    init(engine: ParlorEngine) { self.engine = engine }

    // MARK: errors

    nonisolated static func error(_ status: Int, _ message: String, type: String = "invalid_request_error", code: String? = nil) -> LoopbackServer.Response {
        var e: [String: JSON] = ["message": .string(message), "type": .string(type)]
        if let code { e["code"] = .string(code) }
        return .init(status: status, body: OpenAIWire.encode(.object(["error": .object(e)])))
    }

    private static let unsupportedPaths = ["/v1/embeddings", "/v1/audio", "/v1/completions", "/v1/responses", "/v1/moderations"]

    // MARK: routing

    func handle(_ req: LoopbackServer.Request, caller: Caller) async -> LoopbackServer.Response {
        let path = req.path.split(separator: "?").first.map(String.init) ?? req.path
        switch (req.method, path) {
        case ("GET", "/v1/models"): return models(caller: caller)
        case ("POST", "/v1/chat/completions"): return await chat(req, caller: caller)
        case ("POST", "/v1/images/generations"): return await images(req.body, caller: caller)
        default:
            if Self.unsupportedPaths.contains(where: { path.hasPrefix($0) }) {
                return Self.error(400, "This endpoint is not supported. Parlor serves /v1/models, /v1/chat/completions and /v1/images/generations.", code: "unsupported")
            }
            return Self.error(404, "Unknown route \(req.method) \(path).", code: "not_found")
        }
    }

    // MARK: models

    private func models(caller: Caller) -> LoopbackServer.Response {
        let cat = engine.catalog
        var items: [JSON] = []
        for h in cat.hosts where caller.mayUse(host: h.id) {
            guard let e = cat.engine(h.spec.engine) else { continue }
            let status = cat.status(host: h.id)
            func entry(_ id: String) -> JSON {
                .object(["id": .string(id), "object": .string("model"), "created": .int(0), "owned_by": .string(h.id),
                         "parlor": .object(["host": .string(h.id), "engine": .string(e.id), "capabilities": .array(e.capabilities.map { .string($0) }),
                                            "status": .string(status.label), "auto": .bool(h.isAuto)])])
            }
            if e.remote, let ms = h.spec.models, !ms.isEmpty { items += ms.map { entry(h.id + "/" + $0) } } else { items.append(entry(h.id)) }
        }
        return .init(status: 200, body: OpenAIWire.encode(.object(["object": .string("list"), "data": .array(items)])))
    }

    // MARK: resolving a request's model

    struct ChatTarget { var host: HostInfo; var engine: EngineManifest; var model: String?; var synthetic = false }

    /// A host id, or `<host>/<model>` on a remote host. A model repo is not a reference: agents and clients name hosts.
    func resolveChat(_ model: String) -> ChatTarget? {
        let cat = engine.catalog
        if let h = cat.host(model), let e = cat.engine(h.spec.engine), e.capabilities.contains("chat") {
            if e.remote { return h.spec.models?.first.map { ChatTarget(host: h, engine: e, model: $0) } }
            return ChatTarget(host: h, engine: e, model: nil)
        }
        if let slash = model.firstIndex(of: "/"), let h = cat.host(String(model[..<slash])), let e = cat.engine(h.spec.engine), e.remote {
            return ChatTarget(host: h, engine: e, model: String(model[model.index(after: slash)...]))
        }
        // A substitute runtime (tests) serves any name as an in-memory host.
        if engine.mlx == nil, let e = cat.engines.first(where: { $0.adapter == engine.mlxAdapter.id && $0.capabilities.contains("chat") }) {
            return ChatTarget(host: HostInfo(spec: HostSpec(id: model, engine: e.id, model: model), isAuto: true), engine: e, model: nil, synthetic: true)
        }
        return nil
    }

    /// The one in-process way to ask a host for a completion: the host's queue, then its adapter.
    public func chatStream(_ request: LLMRequest, host: String) -> AsyncThrowingStream<LLMEvent, Error> {
        guard let t = resolveChat(host) else {
            return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("No host \"\(host)\".")) }
        }
        return stream(request, t)
    }

    func stream(_ request: LLMRequest, _ t: ChatTarget) -> AsyncThrowingStream<LLMEvent, Error> {
        guard let adapter = engine.adapters[t.engine.adapter] else {
            return AsyncThrowingStream { $0.finish(throwing: ParlorError.message("Engine \(t.engine.id) names the adapter \(t.engine.adapter), which this app does not have.")) }
        }
        let queues = engine.queues
        let key = t.host.id
        let label = t.model ?? t.host.spec.model ?? t.host.id
        return AsyncThrowingStream { cont in
            let task = Task {
                let queue = await queues.queue(key)
                let ticket: LLMQueue.Ticket
                do { ticket = try await queue.acquire(model: label, priority: request.priority) }
                catch { cont.finish(throwing: error); return }
                do {
                    for try await ev in adapter.chat(request, host: t.host, engine: t.engine, model: t.model) { cont.yield(ev) }
                    await queue.release(ticket)
                    cont.finish()
                } catch {
                    await queue.release(ticket)
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: memory budget

    struct OverBudget: LocalizedError { var errorDescription: String? }

    /// Make room for a host before it starts: stop idle hosts (least recently used first) until it fits the budget.
    func admit(_ host: HostInfo, cost: Int64) async throws {
        guard cost > 0 else { return }
        let limit = Int64(Double(SystemMemory.totalBytes) * engine.settings.memoryBudgetFraction)
        let cat = engine.catalog
        func used() async -> Int64 {
            var total: Int64 = 0
            for a in engine.adapters.all { total += await a.residentBytes() }
            return total
        }
        // Already running: its memory is already counted.
        if let e = cat.engine(host.spec.engine), let a = engine.adapters[e.adapter], await a.usage(of: host).alive { return }
        var current = await used()
        if current + cost <= limit { return }
        var idle: [(host: HostInfo, adapter: Adapter, last: Date)] = []
        for h in cat.hosts where h.id != host.id {
            guard let e = cat.engine(h.spec.engine), let a = engine.adapters[e.adapter], !a.managesOwnBudget else { continue }
            let u = await a.usage(of: h)
            if u.alive, u.active == 0 { idle.append((h, a, u.lastUsed ?? .distantPast)) }
        }
        for v in idle.sorted(by: { $0.last < $1.last }) {
            await v.adapter.stop(v.host)
            try? await Task.sleep(nanoseconds: 300_000_000)
            current = await used()
            if current + cost <= limit { return }
        }
        func gb(_ b: Int64) -> String { String(format: "%.1f GB", Double(b) / 1e9) }
        throw OverBudget(errorDescription: "Host \(host.id) needs about \(gb(cost)); \(gb(current)) is in use and the budget is \(gb(limit)). Stop something in System, or raise the memory budget in Tuning.")
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

    private func num(_ j: JSON?) -> Double? {
        switch j { case .double(let d)?: return d; case .int(let i)?: return Double(i); default: return nil }
    }

    private func chat(_ req: LoopbackServer.Request, caller: Caller) async -> LoopbackServer.Response {
        guard let o = (try? JSONDecoder().decode(JSON.self, from: req.body))?.objectValue else {
            return Self.error(400, "The body is not a JSON object.")
        }
        guard let model = o["model"]?.stringValue, !model.isEmpty else { return Self.error(400, "\"model\" is required.", code: "missing_model") }
        guard let messages = OpenAIWire.parseMessages(o["messages"]), !messages.isEmpty else {
            return Self.error(400, "\"messages\" must be a non-empty list of system, user, assistant or tool messages.", code: "bad_messages")
        }
        if let n = o["n"]?.intValue, n > 1 { return Self.error(400, "n > 1 is not supported.", code: "unsupported") }
        if o["logprobs"] == .bool(true) || (o["top_logprobs"]?.intValue ?? 0) > 0 { return Self.error(400, "logprobs are not supported.", code: "unsupported") }
        guard let target = resolveChat(model) else {
            return Self.error(404, "The model \"\(model)\" is not a host or an installed model. GET /v1/models lists what is available.", code: "model_not_found")
        }
        let scopeID = target.host.id
        if !caller.mayUse(host: scopeID) {
            return Self.error(403, "This token may not use host \"\(scopeID)\".", type: "permission_error", code: "host_forbidden")
        }
        if !target.synthetic, let why = engine.catalog.problem(host: target.host.id) {
            return Self.error(503, why, type: "server_error", code: "model_unavailable")
        }
        let cost = target.host.spec.model.flatMap { engine.catalog.model($0)?.bytes } ?? 0
        if engine.adapters[target.engine.adapter]?.managesOwnBudget != true {
            do { try await admit(target.host, cost: cost) } catch { return Self.error(503, error.localizedDescription, type: "server_error", code: "over_budget") }
        }

        // Sampling: the request, else the host's defaults; only what the engine honours is passed on.
        var asked: [String: JSON] = [:]
        for k in ["temperature", "top_p", "min_p", "top_k", "repetition_penalty", "max_tokens"] {
            if let v = (k == "max_tokens" ? (o["max_completion_tokens"] ?? o["max_tokens"]) : o[k]) { asked[k] = v }
        }
        if asked["repetition_penalty"] == nil, let v = o["repeat_penalty"] { asked["repetition_penalty"] = v }
        var merged = target.host.spec.defaults
        for (k, v) in asked { merged[k] = v }
        let honoured = Set(target.engine.params)
        let ignored = merged.keys.filter { !honoured.contains($0) }.sorted()
        var tuning = LLMTuning(settings: engine.settings.llm)
        var temperature = 0.7
        var maxTokens: Int?
        var extra: [String: JSON] = [:]
        for (k, v) in merged where honoured.contains(k) {
            switch k {
            case "temperature": temperature = num(v) ?? temperature
            case "top_p": tuning.settings.topP = num(v) ?? tuning.settings.topP
            case "max_tokens": maxTokens = v.intValue
            default: extra[k] = v
            }
        }
        // Still silent: the machine's LLM settings (Tuning), then the engine's own default.
        if maxTokens == nil, honoured.contains("max_tokens") { maxTokens = engine.settings.llm.maxReplyTokens }
        if extra["repetition_penalty"] == nil, honoured.contains("repetition_penalty"), let rp = engine.settings.llm.repetitionPenalty { extra["repetition_penalty"] = .double(rp) }
        let priority: LLMPriority = { switch req.headers["x-parlor-priority"]?.lowercased() { case "room": return .room; case "background": return .background; default: return .user } }()
        let stream = o["stream"] == .bool(true)
        let includeUsage = o["stream_options"]?.objectValue?["include_usage"] == .bool(true)
        let thinking = o["chat_template_kwargs"]?.objectValue?["enable_thinking"] == .bool(true)
        let request = LLMRequest(messages: messages, tools: OpenAIWire.parseTools(o["tools"]), temperature: temperature,
                                 maxTokens: maxTokens, thinking: thinking, tuning: tuning, priority: priority, sampling: extra)
        let events = self.stream(request, target)
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
        let created = Int(Date().timeIntervalSince1970)
        let parlorInfo: JSON = .object(["host": .string(target.host.id), "ignored": .array(ignored.map { .string($0) })])

        // Pull the first event before answering, so a model that cannot load is a clean 404/503 rather than a broken stream.
        let pump = Pump(events.makeAsyncIterator(), first: nil)
        do { pump.first = try await pump.next() } catch { return Self.failure(error) }

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
            out["parlor"] = parlorInfo
            return .init(status: 200, body: OpenAIWire.encode(.object(out)))
        }

        return LoopbackServer.Response(streaming: "text/event-stream") { writer in
            func send(_ delta: [String: JSON], finish: String? = nil, usage: JSON? = nil, noChoices: Bool = false) async -> Bool {
                var o: [String: JSON] = ["id": .string(id), "object": .string("chat.completion.chunk"), "created": .int(created), "model": .string(model)]
                o["choices"] = noChoices ? .array([]) : .array([.object(["index": .int(0), "delta": .object(delta), "finish_reason": finish.map { .string($0) } ?? .null])])
                if let usage { o["usage"] = usage; o["parlor"] = parlorInfo }
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

    // MARK: images

    struct ImageFailure: LocalizedError {
        var status: Int; var message: String; var type = "invalid_request_error"; var code: String?
        var errorDescription: String? { message }
    }

    /// The one in-process way to draw: pick the host (named, or the caller's image host), check scope and problems, make room
    /// in the memory budget, wait in the host's queue, then let its adapter draw. The `image.generate` tool and
    /// `/v1/images/generations` both come here.
    public func generateImage(_ r: ImageRequest, host requested: String?, caller: Caller) async throws -> ImageResult {
        let cat = engine.catalog
        let hostID: String
        if let m = requested, !m.isEmpty { hostID = m }
        else if let d = await engine.defaultImageHost(for: caller.kind == .agent ? caller.name : nil) { hostID = d }
        else { throw ImageFailure(status: 404, message: "No image model is downloaded. Download one in Models.", code: "model_not_found") }
        guard let host = cat.host(hostID), cat.capabilities(of: host).contains("image") else {
            throw ImageFailure(status: 404, message: "The model \"\(hostID)\" is not an image host. GET /v1/models lists hosts and their capabilities.", code: "model_not_found")
        }
        guard caller.mayUse(host: host.id) else {
            throw ImageFailure(status: 403, message: "This caller may not use host \"\(host.id)\".", type: "permission_error", code: "host_forbidden")
        }
        if let why = cat.problem(host: host.id) { throw ImageFailure(status: 503, message: why, type: "server_error", code: "model_unavailable") }
        guard let e = cat.engine(host.spec.engine), let adapter = engine.adapters[e.adapter] else {
            throw ImageFailure(status: 503, message: "Host \(host.id) names an adapter this app does not have.", type: "server_error")
        }
        let queue = await engine.queues.queue(host.id)
        let ticket = try await queue.acquire(model: host.spec.model ?? host.id, priority: .user)
        do {
            do { try await admit(host, cost: host.spec.model.flatMap { cat.model($0)?.bytes } ?? 0) }
            catch { throw ImageFailure(status: 503, message: error.localizedDescription, type: "server_error", code: "over_budget") }
            let res = try await adapter.image(r, host: host, engine: e)
            await queue.release(ticket)
            return res
        } catch {
            await queue.release(ticket)
            throw error
        }
    }

    private func images(_ body: Data, caller: Caller) async -> LoopbackServer.Response {
        guard let o = (try? JSONDecoder().decode(JSON.self, from: body))?.objectValue else { return Self.error(400, "The body is not a JSON object.") }
        guard let prompt = o["prompt"]?.stringValue, !prompt.isEmpty else { return Self.error(400, "\"prompt\" is required.", code: "missing_prompt") }
        if let n = o["n"]?.intValue, n > 1 { return Self.error(400, "n > 1 is not supported.", code: "unsupported") }
        var r = ImageRequest(prompt: prompt, negative: (o["negative_prompt"] ?? o["negative"])?.stringValue,
                             shape: o["shape"]?.stringValue ?? "square", detailed: o["detail"]?.stringValue == "detailed",
                             seed: o["seed"]?.intValue, steps: o["steps"]?.intValue)
        if let size = o["size"]?.stringValue {
            let p = size.lowercased().split(separator: "x").compactMap { Int($0) }
            if p.count == 2 { r.width = p[0]; r.height = p[1] }
        }
        do {
            let res = try await generateImage(r, host: o["model"]?.stringValue.flatMap { $0.isEmpty ? nil : $0 }, caller: caller)
            var item: [String: JSON] = ["url": .string(res.url.absoluteString), "parlor_path": .string(res.path),
                                        "revised_prompt": .string(res.prompt), "seed": .int(res.seed), "width": .int(res.width), "height": .int(res.height)]
            if o["response_format"]?.stringValue == "b64_json", let d = try? Data(contentsOf: res.url) { item["b64_json"] = .string(d.base64EncodedString()) }
            return .init(status: 200, body: OpenAIWire.encode(.object(["created": .int(Int(Date().timeIntervalSince1970)), "data": .array([.object(item)]),
                                                                       "parlor": .object(["host": .string(res.host), "seconds": .double(res.seconds)])])))
        } catch is CancellationError {
            return Self.error(499, "cancelled")
        } catch let f as ImageFailure {
            return Self.error(f.status, f.message, type: f.type, code: f.code)
        } catch {
            return Self.error(503, error.localizedDescription, type: "server_error", code: "image_failed")
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
