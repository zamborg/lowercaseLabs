import Foundation

/// Tiny frontmatter reader/writer for the flat schema parlor needs:
/// `key: scalar` and `key:` followed by `  - item` lines. Body follows a closing `---`.
public struct Frontmatter: Sendable {
    public var fields: [(key: String, value: Value)]
    public var body: String

    public enum Value: Sendable, Hashable {
        case scalar(String)
        case list([String])
    }

    public init(fields: [(key: String, value: Value)] = [], body: String = "") {
        self.fields = fields
        self.body = body
    }

    public subscript(key: String) -> Value? {
        get { fields.first(where: { $0.key == key })?.value }
        set {
            if let i = fields.firstIndex(where: { $0.key == key }) {
                if let n = newValue { fields[i].value = n } else { fields.remove(at: i) }
            } else if let n = newValue {
                fields.append((key, n))
            }
        }
    }

    public func string(_ key: String) -> String? {
        if case .scalar(let s)? = self[key] { return s }
        return nil
    }
    public func list(_ key: String) -> [String]? {
        switch self[key] {
        case .list(let l)?: return l
        case .scalar(let s)?: return s.isEmpty ? [] : [s]
        default: return nil
        }
    }

    public static func parse(_ text: String) -> Frontmatter {
        var lines = text.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else {
            return Frontmatter(body: text)
        }
        lines.removeFirst()
        var fields: [(String, Value)] = []
        var i = 0
        var closed = false
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces) == "---" { closed = true; i += 1; break }
            i += 1
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let rest = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            if rest.isEmpty {
                var items: [String] = []
                while i < lines.count {
                    let l = lines[i].trimmingCharacters(in: .whitespaces)
                    if l.hasPrefix("- ") { items.append(unquote(String(l.dropFirst(2)))); i += 1 }
                    else if l.hasPrefix("-") && l.count == 1 { i += 1 }
                    else { break }
                }
                fields.append((key, .list(items)))
            } else if rest == "[]" {
                fields.append((key, .list([])))
            } else {
                fields.append((key, .scalar(unquote(rest))))
            }
        }
        guard closed else { return Frontmatter(body: text) }
        var body = lines[i...].joined(separator: "\n")
        if body.hasPrefix("\n") { body.removeFirst() }
        return Frontmatter(fields: fields, body: body.trimmingCharacters(in: .newlines))
    }

    public func render() -> String {
        var out = "---\n"
        for (k, v) in fields {
            switch v {
            case .scalar(let s): out += "\(k): \(Frontmatter.quote(s))\n"
            case .list(let l):
                if l.isEmpty { out += "\(k): []\n" }
                else {
                    out += "\(k):\n"
                    for item in l { out += "  - \(Frontmatter.quote(item))\n" }
                }
            }
        }
        out += "---\n"
        out += body
        if !body.hasSuffix("\n") { out += "\n" }
        return out
    }

    static func quote(_ s: String) -> String {
        let needs = s.isEmpty || s.first == " " || s.last == " " || s.contains("\n")
            || s.hasPrefix("\"") || s.hasPrefix("'") || s.hasPrefix("#") || s.hasPrefix("- ")
            || s.contains(": ") || s.contains(" #") || s == "[]"
        guard needs else { return s }
        let esc = s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(esc)\""
    }

    static func unquote(_ s: String) -> String {
        guard s.count >= 2, s.hasPrefix("\""), s.hasSuffix("\"") else {
            if s.count >= 2, s.hasPrefix("'"), s.hasSuffix("'") { return String(s.dropFirst().dropLast()) }
            return s
        }
        var out = ""
        var it = s.dropFirst().dropLast().makeIterator()
        while let c = it.next() {
            if c == "\\", let n = it.next() {
                switch n {
                case "n": out.append("\n")
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                default: out.append(n)
                }
            } else { out.append(c) }
        }
        return out
    }
}
