import Foundation

/// One generation at a time per resource. Waiters are ordered by priority, then by whether they use the
/// model that just ran (avoiding a swap), with a fairness bound so nothing waits forever.
public actor LLMQueue {
    private struct Waiter {
        let id: UInt64
        let model: String
        let priority: LLMPriority
        var skips = 0
        let cont: CheckedContinuation<Void, Error>
    }
    public struct Ticket: Sendable { let id: UInt64 }

    private var waiters: [Waiter] = []
    private var running: UInt64?
    private var lastModel: String?
    private var counter: UInt64 = 0
    /// How many times a waiter may be passed over before it must run.
    public let fairness: Int

    public init(fairness: Int = 4) { self.fairness = fairness }

    public var depth: Int { waiters.count }
    public var isBusy: Bool { running != nil }

    public func acquire(model: String, priority: LLMPriority) async throws -> Ticket {
        counter += 1
        let id = counter
        if running == nil, waiters.isEmpty {
            running = id; lastModel = model
            return Ticket(id: id)
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                waiters.append(Waiter(id: id, model: model, priority: priority, cont: c))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        return Ticket(id: id)
    }

    public func release(_ t: Ticket) {
        guard running == t.id else { return }
        running = nil
        startNext()
    }

    private func cancelWaiter(_ id: UInt64) {
        guard let i = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: i).cont.resume(throwing: CancellationError())
    }

    private func startNext() {
        guard running == nil, !waiters.isEmpty else { return }
        let pick: Int
        if let starved = waiters.firstIndex(where: { $0.skips >= fairness }) {
            pick = starved
        } else {
            pick = waiters.indices.min { a, b in
                let x = waiters[a], y = waiters[b]
                if x.priority != y.priority { return x.priority < y.priority }
                let xs = x.model == lastModel, ys = y.model == lastModel
                if xs != ys { return xs }
                return x.id < y.id
            }!
        }
        let w = waiters.remove(at: pick)
        for i in waiters.indices where waiters[i].id < w.id { waiters[i].skips += 1 }
        running = w.id
        lastModel = w.model
        w.cont.resume()
    }
}

/// Puts a queue in front of any runtime, so chats, rooms, summaries and outside clients share one line.
public final class QueuedRuntime: LanguageModelRuntime, @unchecked Sendable {
    public let inner: LanguageModelRuntime
    public let queue: LLMQueue

    public init(inner: LanguageModelRuntime, queue: LLMQueue = LLMQueue()) { self.inner = inner; self.queue = queue }

    public func loadedModel() async -> String? { await inner.loadedModel() }
    public func unload() async { await inner.unload() }

    public func generate(model: String, request: LLMRequest) -> AsyncThrowingStream<LLMEvent, Error> {
        AsyncThrowingStream { cont in
            let task = Task {
                let ticket: LLMQueue.Ticket
                do { ticket = try await queue.acquire(model: model, priority: request.priority) }
                catch { cont.finish(throwing: error); return }
                do {
                    for try await ev in inner.generate(model: model, request: request) { cont.yield(ev) }
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
}
