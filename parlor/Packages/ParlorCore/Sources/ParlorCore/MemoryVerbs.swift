import Foundation

extension CommandCenter {
    /// Which agent's memory a call means. An agent defaults to its own; anything else needs `configure`.
    private func memoryTarget(_ a: Args, _ caller: Caller) throws -> String {
        let own = caller.kind == .agent ? caller.name : nil
        guard let id = a.optString("id") ?? own, !id.isEmpty else { throw CommandError("Missing \"id\": whose memory?") }
        guard engine.agent(id) != nil else { throw CommandError("No agent \"\(id)\".") }
        if id != own, caller.tier < .configure { throw CommandError("Only configure-tier callers can touch another agent's memory.") }
        return id
    }

    func registerMemory() {
        let who: [(String, String, String)] = [("id", "string", "Agent id; an agent's own by default")]
        add("memory.read", .none, "Read an agent's durable memory (memory/<id>.md).", props: who) { a, c in
            let id = try self.memoryTarget(a, c)
            let t = self.engine.memory.read(id)
            return .object(["id": .string(id), "text": .string(t), "chars": .int(t.count), "limit": .int(MemoryStore.maxChars)])
        }
        add("memory.append", .none, "Add a note to an agent's memory. It appears in that agent's system prompt in every conversation.",
            props: who + [("text", "string", "What to remember")], required: ["text"]) { a, c in
            let id = try self.memoryTarget(a, c)
            try self.engine.memory.append(id, try a.string("text"))
            self.engine.noteChange("Memory of \(id)")
            return .object(["id": .string(id), "chars": .int(self.engine.memory.read(id).count)])
        }
        add("memory.replace", .none, "Rewrite an agent's whole memory (empty text clears it). Use it to condense.",
            props: who + [("text", "string", "The new memory")], required: ["text"]) { a, c in
            let id = try self.memoryTarget(a, c)
            let text = a.optString("text") ?? ""
            guard text.count <= MemoryStore.maxChars else { throw CommandError("Memory is limited to \(MemoryStore.maxChars) characters; this is \(text.count).") }
            try self.engine.memory.write(id, text)
            self.engine.noteChange("Memory of \(id)")
            return .object(["id": .string(id), "chars": .int(text.trimmingCharacters(in: .whitespacesAndNewlines).count)])
        }
        add("home.history", .read, "Recent changes to agents, memory, settings and mounts, newest first. Each has a commit hash for home.restore.",
            props: [("limit", "integer", "Default 20"), ("path", "string", "Only changes to this file, e.g. agents/mabel.md")]) { a, _ in
            let entries = self.engine.history.log(limit: a.optInt("limit") ?? 20, path: a.optString("path"))
            let f = ISO8601DateFormatter()
            return .array(entries.map { e in
                .object(["commit": .string(String(e.hash.prefix(10))), "by": .string(e.author), "at": .string(f.string(from: e.date)),
                         "what": .string(e.subject), "files": .strings(e.files)])
            })
        }
        add("home.restore", .configure, destructive: true, "Put files back as they were at a commit from home.history (all tracked files, or just the given paths). Recorded as a new step, so it can be undone too.",
            props: [("commit", "string", "Hash from home.history"), ("paths", "strings", "Files relative to the parlor folder; empty restores everything tracked")], required: ["commit"]) { a, c in
            let changed = try self.engine.history.restore(rev: try a.string("commit"), paths: a.strings("paths") ?? [], actor: c.name)
            self.engine.reloadAgents()
            return .object(["restored": .strings(changed)])
        }
    }
}
