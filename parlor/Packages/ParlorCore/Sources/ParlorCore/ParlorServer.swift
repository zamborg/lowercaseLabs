import Foundation
import Network

/// A loopback-only HTTP/1.1 server that does one job: hand a POST body to a handler and send back its reply.
/// The `parlor` MCP endpoint is built on it (JSON-RPC over HTTP, which streamable-HTTP clients accept when the
/// server answers with plain JSON).
public final class LoopbackServer: @unchecked Sendable {
    public struct Request: Sendable { public var method: String; public var path: String; public var headers: [String: String]; public var body: Data }
    public struct Response: Sendable {
        public var status: Int; public var body: Data; public var contentType = "application/json"
        public init(status: Int, body: Data = Data(), contentType: String = "application/json") { self.status = status; self.body = body; self.contentType = contentType }
    }

    private let queue = DispatchQueue(label: "parlor.server")
    private var listener: NWListener?
    public private(set) var port: UInt16 = 0
    private let handler: @Sendable (Request) async -> Response

    public init(handler: @escaping @Sendable (Request) async -> Response) { self.handler = handler }

    /// Bind to 127.0.0.1. `port` 0 picks a free one. Throws if it cannot listen.
    public func start(port: Int) async throws {
        stop()
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: UInt16(port)) ?? .any)
        let l = try NWListener(using: params)
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        let ready: Void = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let once = OnceBox()
            l.stateUpdateHandler = { state in
                switch state {
                case .ready: if once.take() { cont.resume() }
                case .failed(let e): if once.take() { cont.resume(throwing: ParlorError.message("Could not listen on port \(port): \(e)")) }
                case .cancelled: if once.take() { cont.resume(throwing: ParlorError.message("Listener cancelled")) }
                default: break
                }
            }
            l.start(queue: queue)
        }
        _ = ready
        listener = l
        self.port = l.port?.rawValue ?? UInt16(port)
    }

    public func stop() {
        listener?.cancel(); listener = nil; port = 0
    }

    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        receive(c, Data())
    }

    private func receive(_ c: NWConnection, _ buffer: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, done, error in
            guard let self else { c.cancel(); return }
            var buf = buffer
            if let data { buf.append(data) }
            if error != nil || (done && data == nil) { c.cancel(); return }
            switch Self.parse(buf) {
            case .incomplete:
                if buf.count > 8_000_000 { self.send(c, Response(status: 413)); return }
                self.receive(c, buf)
            case .bad: self.send(c, Response(status: 400))
            case .request(let r):
                Task { let resp = await self.handler(r); self.send(c, resp) }
            }
        }
    }

    private func send(_ c: NWConnection, _ r: Response) {
        let reason = [200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed", 413: "Payload Too Large"][r.status] ?? "OK"
        var head = "HTTP/1.1 \(r.status) \(reason)\r\nContent-Length: \(r.body.count)\r\nConnection: close\r\n"
        if !r.body.isEmpty { head += "Content-Type: \(r.contentType)\r\n" }
        if r.status == 401 { head += "WWW-Authenticate: Bearer\r\n" }
        head += "\r\n"
        var out = Data(head.utf8); out.append(r.body)
        c.send(content: out, completion: .contentProcessed { _ in c.cancel() })
    }

    enum Parsed { case incomplete, bad, request(Request) }

    static func parse(_ d: Data) -> Parsed {
        guard let end = d.range(of: Data("\r\n\r\n".utf8)) else { return .incomplete }
        guard let head = String(data: d[d.startIndex..<end.lowerBound], encoding: .utf8) else { return .bad }
        var lines = head.components(separatedBy: "\r\n")
        let first = lines.removeFirst().split(separator: " ")
        guard first.count >= 2 else { return .bad }
        var headers: [String: String] = [:]
        for l in lines { if let i = l.firstIndex(of: ":") { headers[l[..<i].lowercased()] = l[l.index(after: i)...].trimmingCharacters(in: .whitespaces) } }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = end.upperBound
        guard d.count - (bodyStart - d.startIndex) >= length else { return .incomplete }
        return .request(Request(method: String(first[0]), path: String(first[1]), headers: headers, body: d[bodyStart..<(bodyStart + length)]))
    }
}

final class OnceBox: @unchecked Sendable {
    private let lock = NSLock(); private var done = false
    func take() -> Bool { lock.lock(); defer { lock.unlock() }; if done { return false }; done = true; return true }
}

/// The `parlor` MCP endpoint: tools are the verbs the caller's tier allows.
@MainActor
public final class ParlorMCPEndpoint {
    unowned let engine: ParlorEngine
    public private(set) var server: LoopbackServer?
    /// Secrets of processes the app launched itself (modules, harnesses): tier by secret, gone on quit.
    private var ephemeral: [String: Caller] = [:]

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

    /// A token for a process the app launches. Not stored; it dies with the app.
    public func issueEphemeral(name: String, tier: Tier) -> String {
        let secret = "plt_eph_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        ephemeral[secret] = .token(name, tier)
        return secret
    }

    private func caller(for bearer: String) -> Caller? {
        if let c = ephemeral[bearer] { return c }
        let h = APIToken.hash(bearer)
        guard let t = engine.settings.tokens.first(where: { $0.hash == h }), !t.isExpired else { return nil }
        return .token(t.name, t.tier)
    }

    func handle(_ req: LoopbackServer.Request) async -> LoopbackServer.Response {
        // A browser page must not be able to drive the app: refuse any foreign Origin or Host.
        if let o = req.headers["origin"], !o.isEmpty, !(o.contains("://127.0.0.1") || o.contains("://localhost")) { return .init(status: 403) }
        if let h = req.headers["host"], !(h.hasPrefix("127.0.0.1") || h.hasPrefix("localhost") || h.hasPrefix("[::1]")) { return .init(status: 403) }
        guard req.path == "/mcp" || req.path.hasPrefix("/mcp?") else { return .init(status: 404) }
        guard req.method == "POST" else { return .init(status: 405) }
        let auth = req.headers["authorization"] ?? ""
        guard auth.lowercased().hasPrefix("bearer "), let caller = caller(for: String(auth.dropFirst(7)).trimmingCharacters(in: .whitespaces)) else {
            return .init(status: 401)
        }
        guard let msg = try? JSONDecoder().decode(JSON.self, from: req.body), let obj = msg.objectValue else { return .init(status: 400) }
        let method = obj["method"]?.stringValue ?? ""
        let id = obj["id"]
        guard let id else { return .init(status: 202) }   // a notification
        func reply(_ result: JSON) -> LoopbackServer.Response {
            let body = JSON.object(["jsonrpc": .string("2.0"), "id": id, "result": result])
            return .init(status: 200, body: (try? JSONEncoder().encode(body)) ?? Data())
        }
        func fail(_ code: Int, _ message: String) -> LoopbackServer.Response {
            let body = JSON.object(["jsonrpc": .string("2.0"), "id": id, "error": .object(["code": .int(code), "message": .string(message)])])
            return .init(status: 200, body: (try? JSONEncoder().encode(body)) ?? Data())
        }
        let params = obj["params"]?.objectValue ?? [:]
        switch method {
        case "initialize":
            return reply(.object(["protocolVersion": params["protocolVersion"] ?? .string("2025-03-26"),
                                  "capabilities": .object(["tools": .object(["listChanged": .bool(false)])]),
                                  "serverInfo": .object(["name": .string("parlor"), "version": .string("0.2.0")]),
                                  "instructions": .string("Controls for the parlor app. Your tier (\(caller.tier.rawValue)) decides which tools are listed; destructive ones ask the user first.")]))
        case "ping": return reply(.object([:]))
        case "tools/list":
            let tools = engine.commands.sorted.filter { $0.tier <= caller.tier }.map { v -> JSON in
                .object(["name": .string(v.name), "description": .string("[\(v.tier.rawValue)\(v.destructive ? ", asks first" : "")] " + v.summary), "inputSchema": v.schema])
            }
            return reply(.object(["tools": .array(tools)]))
        case "tools/call":
            guard let name = params["name"]?.stringValue else { return fail(-32602, "Missing tool name") }
            let out = await engine.commands.call(name, params["arguments"] ?? .object([:]), caller: caller)
            return reply(.object(["content": .array([.object(["type": .string("text"), "text": .string(out.text)])]), "isError": .bool(out.isError)]))
        default: return fail(-32601, "Method not found: \(method)")
        }
    }
}
