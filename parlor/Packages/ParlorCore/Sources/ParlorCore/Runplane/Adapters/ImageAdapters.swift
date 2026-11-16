import Foundation

/// The image service is created when the engine starts; adapters read it through this box.
final class ImageBox: @unchecked Sendable {
    private let lock = NSLock()
    private var p: ImageProvider?
    var provider: ImageProvider? {
        get { lock.lock(); defer { lock.unlock() }; return p }
        set { lock.lock(); p = newValue; lock.unlock() }
    }
}

/// `command`: one process per image (sd-cli and friends). The work is `ImageProvider`'s; the adapter is the way in.
public final class CommandImageAdapter: Adapter, @unchecked Sendable {
    public let id = "command"
    private let box: ImageBox
    init(image: ImageBox) { box = image }

    public func image(_ req: ImageRequest, host: HostInfo, engine: EngineManifest) async throws -> ImageResult {
        guard let p = box.provider else { throw ParlorError.message("The image service is not running.") }
        return try await p.generate(host: host.id, req)
    }
}

/// `mcp-stdio`: a long-lived worker that speaks MCP over stdio (the Z-Image worker).
public final class MCPImageAdapter: Adapter, @unchecked Sendable {
    public let id = "mcp-stdio"
    private let box: ImageBox
    init(image: ImageBox) { box = image }

    public func image(_ req: ImageRequest, host: HostInfo, engine: EngineManifest) async throws -> ImageResult {
        guard let p = box.provider else { throw ParlorError.message("The image service is not running.") }
        return try await p.generate(host: host.id, req)
    }

    public func usage(of host: HostInfo) async -> HostUsage {
        guard let p = box.provider, let u = await p.workerUsage(host: host.id) else { return HostUsage(alive: false) }
        return HostUsage(alive: true, active: u.active, lastUsed: u.lastUsed, bytes: u.bytes)
    }

    public func residentBytes() async -> Int64 {
        guard let p = box.provider else { return 0 }
        var t: Int64 = 0
        for pid in await p.workerPIDs() { if let b = ProcessMemory.footprint(pid: pid) { t += Int64(b) } }
        return t
    }

    public func start(_ host: HostInfo) async throws { try await box.provider?.start() }
    public func stop(_ host: HostInfo) async { await box.provider?.stop(host: host.id) }
    public func shutdown() async { await box.provider?.shutdown() }
}
