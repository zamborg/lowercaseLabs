import Foundation

/// A problem a model (or a person) can fix: where it is, what is wrong, and what to try.
public struct Issue: Codable, Equatable, Sendable {
    public enum Severity: String, Codable, Sendable { case error, warning }
    public var code: String
    public var path: String
    public var message: String
    public var hint: String?
    public var severity: Severity = .error

    public init(code: String = "invalid", path: String, message: String, hint: String? = nil, severity: Severity = .error) {
        self.code = code; self.path = path; self.message = message; self.hint = hint; self.severity = severity
    }

    public var json: JSON {
        var o: [String: JSON] = ["code": .string(code), "path": .string(path), "message": .string(message), "severity": .string(severity.rawValue)]
        if let hint { o["hint"] = .string(hint) }
        return .object(o)
    }
}

public extension Array where Element == Issue {
    var errors: [Issue] { filter { $0.severity == .error } }
    var json: JSON { .array(map(\.json)) }
    /// One readable block for a thrown error.
    var summary: String { map { "\($0.path.isEmpty ? "" : $0.path + ": ")\($0.message)" + ($0.hint.map { " (\($0))" } ?? "") }.joined(separator: "\n") }
}

/// Thrown by create/update when `validate` finds errors; carries all of them.
public struct ValidationError: LocalizedError, Sendable {
    public var issues: [Issue]
    public var errorDescription: String? { issues.errors.summary }
}

// MARK: - the schema subset

/// The JSON Schema subset parlor understands, used for harness `config`, engine `settings` and action `args`:
/// `type` (string, number, integer, boolean, array of string, object), `enum`, `default`, `minimum`/`maximum`,
/// `format: multiline` and `description`.
public enum SchemaCheck {
    public static func validate(_ value: JSON, schema: JSON, path: String = "") -> [Issue] {
        guard let s = schema.objectValue else { return [] }
        var out: [Issue] = []
        if case .array(let options)? = s["enum"] {
            if !options.contains(value) {
                let names = options.map { $0.stringValue ?? $0.text }.joined(separator: ", ")
                out.append(Issue(path: path, message: "must be one of \(names)"))
            }
            return out
        }
        switch s["type"]?.stringValue {
        case "string": if value.stringValue == nil { out.append(Issue(path: path, message: "must be a string")) }
        case "boolean": if case .bool = value {} else { out.append(Issue(path: path, message: "must be true or false")) }
        case "integer", "number":
            let n: Double?
            switch value { case .int(let i): n = Double(i); case .double(let d): n = d; default: n = nil }
            if let n {
                if s["type"]?.stringValue == "integer", n != n.rounded() { out.append(Issue(path: path, message: "must be a whole number")) }
                if let lo = number(s["minimum"]), n < lo { out.append(Issue(path: path, message: "must be at least \(s["minimum"]!.text)")) }
                if let hi = number(s["maximum"]), n > hi { out.append(Issue(path: path, message: "must be at most \(s["maximum"]!.text)")) }
            } else { out.append(Issue(path: path, message: "must be a number")) }
        case "array":
            if case .array(let items) = value {
                if let itemSchema = s["items"] { for (i, v) in items.enumerated() { out += validate(v, schema: itemSchema, path: "\(path)[\(i)]") } }
            } else { out.append(Issue(path: path, message: "must be a list")) }
        case "object":
            guard let o = value.objectValue else { out.append(Issue(path: path, message: "must be an object")); break }
            let props = s["properties"]?.objectValue ?? [:]
            for (k, v) in o {
                let p = path.isEmpty ? k : path + "." + k
                if let ps = props[k] { out += validate(v, schema: ps, path: p) }
                else { out.append(Issue(code: "unknown", path: p, message: "not a known setting", hint: "known: " + props.keys.sorted().joined(separator: ", "), severity: .warning)) }
            }
        default: break
        }
        return out
    }

    /// Schema defaults for every property that has one.
    public static func defaults(_ schema: JSON?) -> [String: JSON] {
        guard let props = schema?.objectValue?["properties"]?.objectValue else { return [:] }
        var out: [String: JSON] = [:]
        for (k, v) in props { if let d = v.objectValue?["default"] { out[k] = d } }
        return out
    }

    private static func number(_ j: JSON?) -> Double? {
        switch j { case .int(let i)?: return Double(i); case .double(let d)?: return d; default: return nil }
    }
}

// MARK: - nouns

/// Anything with a JSON file: engine, host, harness, agent. One Codable struct is the file, the verb input and output,
/// and the wire JSON.
public protocol Noun: Codable, Sendable {
    static var kind: String { get }
    var id: String { get set }
    /// Extra checks beyond "it decodes". `draft` is the full JSON the caller meant.
    static func check(_ draft: [String: JSON]) -> [Issue]
}

public extension Noun {
    static func check(_ draft: [String: JSON]) -> [Issue] { [] }
}

public enum NounLayout: Sendable {
    /// `<dir>/<id>.json`
    case file
    /// `<dir>/<id>/<name>`
    case folder(String)
}

/// The only reader and writer of noun files. Unknown keys in a file survive an update.
public final class NounStore<T: Noun>: @unchecked Sendable {
    public let dir: URL
    public let layout: NounLayout

    public init(dir: URL, layout: NounLayout) { self.dir = dir; self.layout = layout }

    public static func slugOK(_ id: String) -> Bool {
        !id.isEmpty && id.count <= 64 && id.unicodeScalars.allSatisfy { ($0.isASCII && CharacterSet.alphanumerics.contains($0)) || $0 == "-" || $0 == "_" || $0 == "." }
            && !id.hasPrefix(".")
    }

    public func url(_ id: String) -> URL {
        switch layout {
        case .file: return dir.appendingPathComponent(id + ".json")
        case .folder(let name): return dir.appendingPathComponent(id).appendingPathComponent(name)
        }
    }

    public func ids() -> [String] {
        let fm = FileManager.default
        let names = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).sorted()
        switch layout {
        case .file:
            return names.filter { $0.hasSuffix(".json") && !$0.hasPrefix(".") && !$0.hasSuffix(".bench.json") }.map { String($0.dropLast(5)) }
        case .folder(let file):
            return names.filter { !$0.hasPrefix(".") && fm.fileExists(atPath: dir.appendingPathComponent($0).appendingPathComponent(file).path) }
        }
    }

    public func raw(_ id: String) -> [String: JSON]? {
        guard let d = try? Data(contentsOf: url(id)), let j = try? JSONDecoder().decode(JSON.self, from: d) else { return nil }
        return j.objectValue
    }

    public func read(_ id: String) -> T? {
        guard let r = raw(id), let data = try? JSONEncoder().encode(JSON.object(r)) else { return nil }
        guard var v = try? JSONDecoder().decode(T.self, from: data) else { return nil }
        v.id = id
        return v
    }

    public func list() -> [T] { ids().compactMap { read($0) } }

    /// Report every problem with a draft, writing nothing.
    public func validate(_ draft: JSON) -> [Issue] {
        guard let o = draft.objectValue else { return [Issue(path: "", message: "must be a JSON object")] }
        var out: [Issue] = []
        if let id = o["id"]?.stringValue {
            if !Self.slugOK(id) { out.append(Issue(path: "id", message: "use letters, digits, - and _ only", hint: "for example \"\(Agent.slug(id))\"")) }
        } else { out.append(Issue(code: "missing", path: "id", message: "required")) }
        if let data = try? JSONEncoder().encode(draft) {
            do { _ = try JSONDecoder().decode(T.self, from: data) }
            catch let e as DecodingError { out.append(Self.issue(e)) }
            catch { out.append(Issue(path: "", message: error.localizedDescription)) }
        }
        out += T.check(o)
        return out
    }

    static func issue(_ e: DecodingError) -> Issue {
        func p(_ c: [CodingKey]) -> String { c.map(\.stringValue).joined(separator: ".") }
        switch e {
        case .keyNotFound(let k, let c): return Issue(code: "missing", path: p(c.codingPath + [k]), message: "required")
        case .typeMismatch(_, let c): return Issue(path: p(c.codingPath), message: "wrong type")
        case .valueNotFound(_, let c): return Issue(code: "missing", path: p(c.codingPath), message: "required")
        case .dataCorrupted(let c): return Issue(path: p(c.codingPath), message: c.debugDescription)
        @unknown default: return Issue(path: "", message: "could not be read")
        }
    }

    @discardableResult
    public func create(_ draft: JSON) throws -> T {
        let issues = validate(draft)
        if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
        let id = draft.objectValue!["id"]!.stringValue!
        if FileManager.default.fileExists(atPath: url(id).path) {
            throw ValidationError(issues: [Issue(code: "exists", path: "id", message: "\(T.kind) \"\(id)\" already exists", hint: "use update, or pick another id")])
        }
        try write(id, draft.objectValue!)
        return read(id)!
    }

    /// Save a typed value, keeping any keys in the file that the type does not know about.
    @discardableResult
    public func save(_ value: T) throws -> T {
        let data = try JSONEncoder().encode(value)
        guard let o = try JSONDecoder().decode(JSON.self, from: data).objectValue else { return value }
        var merged = raw(value.id) ?? [:]
        for (k, v) in o { merged[k] = v }
        try write(value.id, merged)
        return read(value.id) ?? value
    }

    /// JSON merge patch (RFC 7386): objects merge, `null` removes a key, anything else replaces.
    @discardableResult
    public func update(_ id: String, patch: JSON) throws -> T {
        guard let current = raw(id) else {
            throw ValidationError(issues: [Issue(code: "missing", path: "id", message: "no \(T.kind) \"\(id)\"", hint: "available: " + ids().joined(separator: ", "))])
        }
        var merged = Self.merge(.object(current), patch).objectValue ?? current
        merged["id"] = .string(id)
        let issues = validate(.object(merged))
        if !issues.errors.isEmpty { throw ValidationError(issues: issues) }
        try write(id, merged)
        return read(id)!
    }

    public func delete(_ id: String) throws {
        switch layout {
        case .file: try FileManager.default.removeItem(at: url(id))
        case .folder: try FileManager.default.removeItem(at: dir.appendingPathComponent(id))
        }
    }

    func write(_ id: String, _ object: [String: JSON]) throws {
        let u = url(id)
        try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(JSON.object(object)).write(to: u, options: .atomic)
    }

    public static func merge(_ base: JSON, _ patch: JSON) -> JSON {
        guard case .object(let p) = patch else { return patch }
        var out = base.objectValue ?? [:]
        for (k, v) in p {
            if case .null = v { out[k] = nil } else { out[k] = merge(out[k] ?? .null, v) }
        }
        return .object(out)
    }
}
