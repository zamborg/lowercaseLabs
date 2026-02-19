import Foundation

struct EmotionTagDefinition: Identifiable, Hashable {
    let id: String
    let pleasantness: Double
    let energy: Double
    let aliases: [String]
}

enum EmotionTaxonomy {
    static let tags: [EmotionTagDefinition] = [
        tag("ecstatic", pleasantness: 0.95, energy: 0.95, aliases: ["elated", "thrilled"]),
        tag("excited", pleasantness: 0.82, energy: 0.90),
        tag("joyful", pleasantness: 0.85, energy: 0.78, aliases: ["happy"]),
        tag("playful", pleasantness: 0.68, energy: 0.74),
        tag("inspired", pleasantness: 0.74, energy: 0.80),
        tag("proud", pleasantness: 0.64, energy: 0.68),
        tag("optimistic", pleasantness: 0.72, energy: 0.63),
        tag("hopeful", pleasantness: 0.66, energy: 0.56),
        tag("grateful", pleasantness: 0.58, energy: 0.44, aliases: ["thankful"]),
        tag("confident", pleasantness: 0.70, energy: 0.52),
        tag("motivated", pleasantness: 0.63, energy: 0.68, aliases: ["driven"]),
        tag("energized", pleasantness: 0.55, energy: 0.90, aliases: ["energetic"]),
        tag("focused", pleasantness: 0.50, energy: 0.47),
        tag("curious", pleasantness: 0.42, energy: 0.63),
        tag("amused", pleasantness: 0.56, energy: 0.58),
        tag("loving", pleasantness: 0.80, energy: 0.42),
        tag("connected", pleasantness: 0.60, energy: 0.38, aliases: ["connection"]),
        tag("supported", pleasantness: 0.52, energy: 0.30),

        tag("anxious", pleasantness: -0.68, energy: 0.76, aliases: ["social anxiety", "worried"]),
        tag("stressed", pleasantness: -0.62, energy: 0.66, aliases: ["stress", "work pressure", "school pressure", "family stress", "financial stress"]),
        tag("overwhelmed", pleasantness: -0.72, energy: 0.58),
        tag("restless", pleasantness: -0.42, energy: 0.52),
        tag("frustrated", pleasantness: -0.58, energy: 0.48, aliases: ["relationship tension"]),
        tag("irritated", pleasantness: -0.50, energy: 0.42, aliases: ["annoyed"]),
        tag("angry", pleasantness: -0.76, energy: 0.66, aliases: ["mad"]),
        tag("furious", pleasantness: -0.88, energy: 0.82),
        tag("resentful", pleasantness: -0.70, energy: 0.45),
        tag("jealous", pleasantness: -0.55, energy: 0.50),
        tag("panicked", pleasantness: -0.92, energy: 0.93, aliases: ["panic"]),
        tag("afraid", pleasantness: -0.74, energy: 0.72, aliases: ["fearful"]),
        tag("nervous", pleasantness: -0.55, energy: 0.62),
        tag("embarrassed", pleasantness: -0.34, energy: 0.48),

        tag("calm", pleasantness: 0.62, energy: -0.22, aliases: ["self care", "self-care"]),
        tag("content", pleasantness: 0.56, energy: -0.30),
        tag("relaxed", pleasantness: 0.66, energy: -0.42),
        tag("peaceful", pleasantness: 0.74, energy: -0.55),
        tag("secure", pleasantness: 0.42, energy: -0.35),
        tag("balanced", pleasantness: 0.46, energy: -0.18),
        tag("satisfied", pleasantness: 0.52, energy: -0.15),
        tag("relieved", pleasantness: 0.38, energy: -0.26),
        tag("cozy", pleasantness: 0.50, energy: -0.52),
        tag("mindful", pleasantness: 0.45, energy: -0.12, aliases: ["reflection", "reflecting"]),
        tag("okay", pleasantness: 0.12, energy: -0.06),
        tag("reflective", pleasantness: 0.18, energy: -0.28, aliases: ["reflection"]),

        tag("tired", pleasantness: -0.32, energy: -0.42, aliases: ["sleep issues", "health worry"]),
        tag("drained", pleasantness: -0.46, energy: -0.55, aliases: ["exhausted"]),
        tag("sleepy", pleasantness: -0.12, energy: -0.68),
        tag("bored", pleasantness: -0.24, energy: -0.36),
        tag("numb", pleasantness: -0.36, energy: -0.72),
        tag("sad", pleasantness: -0.72, energy: -0.44, aliases: ["down"]),
        tag("disappointed", pleasantness: -0.58, energy: -0.28),
        tag("discouraged", pleasantness: -0.66, energy: -0.36),
        tag("lonely", pleasantness: -0.80, energy: -0.52, aliases: ["alone", "isolated"]),
        tag("hopeless", pleasantness: -0.90, energy: -0.68),
        tag("guilty", pleasantness: -0.52, energy: -0.40),
        tag("ashamed", pleasantness: -0.64, energy: -0.48),
        tag("insecure", pleasantness: -0.48, energy: -0.38, aliases: ["self doubt", "self-doubt"]),
        tag("grief", pleasantness: -0.86, energy: -0.62),
        tag("depressed", pleasantness: -0.94, energy: -0.78),
    ]

    private static let tagsByID: [String: EmotionTagDefinition] = {
        Dictionary(uniqueKeysWithValues: tags.map { ($0.id, $0) })
    }()

    private static let aliasToTag: [String: String] = {
        var map: [String: String] = [:]
        for tag in tags {
            map[normalize(tag.id)] = tag.id
            for alias in tag.aliases {
                map[normalize(alias)] = tag.id
            }
        }
        map[normalize("exercise")] = "energized"
        map[normalize("growth")] = "motivated"
        return map
    }()

    static func definition(for tag: String) -> EmotionTagDefinition? {
        let normalized = normalize(tag)
        guard let canonical = aliasToTag[normalized] else {
            return nil
        }
        return tagsByID[canonical]
    }

    static func canonicalTag(for tag: String) -> String {
        definition(for: tag)?.id ?? normalize(tag)
    }

    static func displayName(for tag: String) -> String {
        let canonical = definition(for: tag)?.id ?? canonicalTag(for: tag)
        return canonical
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    static func matchedTags(
        in _: String,
        maxCount: Int,
        phraseMatcher: (String) -> Bool
    ) -> [String] {
        var matches: [(tag: EmotionTagDefinition, score: Int, index: Int)] = []

        for (index, tag) in tags.enumerated() {
            let terms = [tag.id] + tag.aliases
            let score = terms.reduce(0) { partial, term in
                partial + (phraseMatcher(term) ? 1 : 0)
            }
            if score > 0 {
                matches.append((tag: tag, score: score, index: index))
            }
        }

        if matches.isEmpty {
            return ["reflective"]
        }

        let sorted = matches.sorted {
            if $0.score != $1.score {
                return $0.score > $1.score
            }
            return $0.index < $1.index
        }

        var unique: [String] = []
        for match in sorted {
            if unique.contains(match.tag.id) {
                continue
            }
            unique.append(match.tag.id)
            if unique.count >= max(1, maxCount) {
                break
            }
        }
        return unique
    }

    static func moodScore(for tags: [String]) -> Double {
        let coordinates = tags.compactMap { definition(for: $0)?.pleasantness }
        guard !coordinates.isEmpty else {
            return 0.0
        }
        let average = coordinates.reduce(0, +) / Double(coordinates.count)
        return max(-2.0, min(2.0, average * 2.0))
    }

    static func averageEnergy(for tags: [String]) -> Double {
        let coordinates = tags.compactMap { definition(for: $0)?.energy }
        guard !coordinates.isEmpty else {
            return 0.0
        }
        return coordinates.reduce(0, +) / Double(coordinates.count)
    }

    static func averagePleasantness(for tags: [String]) -> Double {
        let coordinates = tags.compactMap { definition(for: $0)?.pleasantness }
        guard !coordinates.isEmpty else {
            return 0.0
        }
        return coordinates.reduce(0, +) / Double(coordinates.count)
    }

    static func normalize(_ value: String) -> String {
        let base = value
            .lowercased()
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let parts = base.split(whereSeparator: { $0.isWhitespace })
        return parts.joined(separator: " ")
    }

    private static func tag(
        _ id: String,
        pleasantness: Double,
        energy: Double,
        aliases: [String] = []
    ) -> EmotionTagDefinition {
        EmotionTagDefinition(
            id: normalize(id),
            pleasantness: max(-1.0, min(1.0, pleasantness)),
            energy: max(-1.0, min(1.0, energy)),
            aliases: aliases.map(normalize)
        )
    }
}
