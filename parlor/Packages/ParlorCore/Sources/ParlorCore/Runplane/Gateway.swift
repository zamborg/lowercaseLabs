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

    struct ChatTarget { var route: String; var host: HostInfo?; var engine: EngineManifest? }

    /// A host id, `<host>/<model>` on a remote host, or a model repo.
    func resolveChat(_ model: String) -> ChatTarget? {
        let cat = engine.catalog
        if let h = cat.host(model), let e = cat.engine(h.spec.engine), e.capabilities.contains("chat") {
            switch e.adapter {
            case "inprocess-mlx": return h.spec.model.map { ChatTarget(route: $0, host: h, engine: e) }
            default:
                if e.remote {
                    let first = h.spec.models?.first
                    return first.map { ChatTarget(route: h.id + "/" + $0, host: h, engine: e) }
                }
                return ChatTarget(route: h.id + "/" + h.id, host: h, engine: e)
            }
        }
        if let slash = model.firstIndex(of: "/"), let h = cat.host(String(model[..<slash])), let e = cat.engine(h.spec.engine), e.remote {
            return ChatTarget(route: model, host: h, engine: e)
        }
        guard engine.runtimeKnowsModel(model) else { return nil }
        let h = cat.resolve(model)
        return ChatTarget(route: model, host: h, engine: h.flatMap { cat.engine($0.spec.engine) })
    }

    // MARK: memory budget

    struct OverBudget: LocalizedError { var errorDescription: String? }

    /// Make room for a host before it starts: stop idle hosts (least recently used first) until it fits the budget.
    func admit(_ host: HostInfo?, cost: Int64) async throws {
        guard let host, cost > 0 else { return }
        let limit = Int64(Double(SystemMemory.totalBytes) * engine.settings.memoryBudgetFraction)
        func used() async -> Int64 {
            var total: Int64 = 0
            for p in engine.router.all {
                if let s = p.process?.snapshot(), s.state.isAlive, let pid = s.pid, let b = ProcessMemory.footprint(pid: pid) { total += Int64(b) }
            }
            for pid in await engine.image?.workerPIDs() ?? [] { if let b = ProcessMemory.footprint(pid: pid) { total += Int64(b) } }
            if let m = engine.mlx { total += Int64(max(0, await m.residentBytes())) }
            return total
        }
        // Already running: its memory is already counted.
        if let s = engine.router.provider(host.id)?.process?.snapshot(), s.state.isAlive { return }
        if await engine.image?.hasWorker(host.id) == true { return }
        var current = await used()
        if current + cost <= limit { return }
        let idle = engine.router.all.compactMap { p -> (String, Date)? in
            guard p.config.id != host.id, let s = p.process?.snapshot(), s.state.isAlive, s.active == 0 else { return nil }
            return (p.config.id, s.lastUsed ?? .distantPast)
        }.sorted { $0.1 < $1.1 }
        for (id, _) in idle {
            engine.router.provider(id)?.process?.stop()
            try? await Task.sleep(nanoseconds: 300_000_000)
            current = await used()
            if current + cost <= limit { return }
        }
        for id in await engine.image?.idleWorkers() ?? [] where id != host.id {
            await engine.image?.stop(host: id)
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
        let scopeID = target.host?.id ?? engine.router.queueKey(target.route)
        if !caller.mayUse(host: scopeID) {
            return Self.error(403, "This token may not use host \"\(scopeID)\".", type: "permission_error", code: "host_forbidden")
        }
        if let h = target.host, let why = engine.catalog.problem(host: h.id) {
            return Self.error(503, why, type: "server_error", code: "model_unavailable")
        }
        let cost = target.host.flatMap { h in h.spec.model.flatMap { engine.catalog.model($0)?.bytes } } ?? 0
        if target.engine?.adapter != "inprocess-mlx" {
            do { try await admit(target.host, cost: cost) } catch { return Self.error(503, error.localizedDescription, type: "server_error", code: "over_budget") }
        }

        // Sampling: the request, else the host's defaults; only what the engine honours is passed on.
        var asked: [String: JSON] = [:]
        for k in ["temperature", "top_p", "min_p", "top_k", "repetition_penalty", "max_tokens"] {
            if let v = (k == "max_tokens" ? (o["max_completion_tokens"] ?? o["max_tokens"]) : o[k]) { asked[k] = v }
        }
        if asked["repetition_penalty"] == nil, let v = o["repeat_penalty"] { asked["repetition_penalty"] = v }
        var merged = target.host?.spec.defaults ?? [:]
        for (k, v) in asked { merged[k] = v }
        let honoured = Set(target.engine?.params ?? ["temperature", "top_p", "max_tokens"])
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
        let priority: LLMPriority = { switch req.headers["x-parlor-priority"]?.lowercased() { case "room": return .room; case "background": return .background; default: return .user } }()
        let stream = o["stream"] == .bool(true)
        let includeUsage = o["stream_options"]?.objectValue?["include_usage"] == .bool(true)
        let thinking = o["chat_template_kwargs"]?.objectValue?["enable_thinking"] == .bool(true)
        let request = LLMRequest(messages: messages, tools: OpenAIWire.parseTools(o["tools"]), temperature: temperature,
                                 maxTokens: maxTokens, thinking: thinking, tuning: tuning, priority: priority, sampling: extra)
        let events = engine.runtime.generate(model: target.route, request: request)
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
        let created = Int(Date().timeIntervalSince1970)
        let parlorInfo: JSON = .object(["host": .string(target.host?.id ?? ""), "ignored": .array(ignored.map { .string($0) })])

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

    private func images(_ body: Data, caller: Caller) async -> LoopbackServer.Response {
        guard let o = (try? JSONDecoder().decode(JSON.self, from: body))?.objectValue else { return Self.error(400, "The body is not a JSON object.") }
        guard let prompt = o["prompt"]?.stringValue, !prompt.isEmpty else { return Self.error(400, "\"prompt\" is required.", code: "missing_prompt") }
        if let n = o["n"]?.intValue, n > 1 { return Self.error(400, "n > 1 is not supported.", code: "unsupported") }
        let cat = engine.catalog
        let hostID: String
        if let m = o["model"]?.stringValue, !m.isEmpty { hostID = m }
        else if let d = await engine.defaultImageHost(for: nil) { hostID = d }
        else { return Self.error(404, "No image host is available. GET /v1/models lists hosts.", code: "model_not_found") }
        guard let host = cat.host(hostID), cat.capabilities(of: host).contains("image") else {
            return Self.error(404, "The model \"\(hostID)\" is not an image host. GET /v1/models lists hosts and their capabilities.", code: "model_not_found")
        }
        guard caller.mayUse(host: host.id) else {
            return Self.error(403, "This token may not use host \"\(host.id)\".", type: "permission_error", code: "host_forbidden")
        }
        if let why = cat.problem(host: host.id) { return Self.error(503, why, type: "server_error", code: "model_unavailable") }
        do { try await admit(host, cost: host.spec.model.flatMap { cat.model($0)?.bytes } ?? 0) }
        catch { return Self.error(503, error.localizedDescription, type: "server_error", code: "over_budget") }
        var r = ImageRequest(prompt: prompt, negative: (o["negative_prompt"] ?? o["negative"])?.stringValue,
                             shape: o["shape"]?.stringValue ?? "square", detailed: o["detail"]?.stringValue == "detailed",
                             seed: o["seed"]?.intValue, steps: o["steps"]?.intValue)
        if let size = o["size"]?.stringValue {
            let p = size.lowercased().split(separator: "x").compactMap { Int($0) }
            if p.count == 2 { r.width = p[0]; r.height = p[1] }
        }
        guard let image = engine.image else { return Self.error(503, "The image service is not running.", type: "server_error") }
        do {
            let res = try await image.generate(host: host.id, r)
            var item: [String: JSON] = ["url": .string(res.url.absoluteString), "parlor_path": .string(res.path),
                                        "revised_prompt": .string(res.prompt), "seed": .int(res.seed), "width": .int(res.width), "height": .int(res.height)]
            if o["response_format"]?.stringValue == "b64_json", let d = try? Data(contentsOf: res.url) { item["b64_json"] = .string(d.base64EncodedString()) }
            return .init(status: 200, body: OpenAIWire.encode(.object(["created": .int(Int(Date().timeIntervalSince1970)), "data": .array([.object(item)]),
                                                                       "parlor": .object(["host": .string(host.id), "seconds": .double(res.seconds)])])))
        } catch is CancellationError {
            return Self.error(499, "cancelled")
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
