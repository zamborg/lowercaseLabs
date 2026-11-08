import Foundation

public struct WordErrorScore: Codable, Sendable, Equatable {
    public let referenceWordCount: Int
    public let hypothesisWordCount: Int
    public let substitutions: Int
    public let deletions: Int
    public let insertions: Int
    public let errorCount: Int
    /// Undefined for an empty reference; error counts are still available.
    public let wer: Double?

    public init(referenceWordCount: Int, hypothesisWordCount: Int,
                substitutions: Int, deletions: Int, insertions: Int) {
        self.referenceWordCount = referenceWordCount
        self.hypothesisWordCount = hypothesisWordCount
        self.substitutions = substitutions
        self.deletions = deletions
        self.insertions = insertions
        errorCount = substitutions + deletions + insertions
        wer = referenceWordCount > 0 ? Double(errorCount) / Double(referenceWordCount) : nil
    }

    /// Corpus WER uses total errors / total reference words, not mean clip WER.
    public static func aggregate(_ scores: [Self]) -> Self {
        Self(referenceWordCount: scores.reduce(0) { $0 + $1.referenceWordCount },
             hypothesisWordCount: scores.reduce(0) { $0 + $1.hypothesisWordCount },
             substitutions: scores.reduce(0) { $0 + $1.substitutions },
             deletions: scores.reduce(0) { $0 + $1.deletions },
             insertions: scores.reduce(0) { $0 + $1.insertions })
    }
}

public enum WordErrorRate {
    /// Canonical Unicode composition, fixed en-US lowercase, remove punctuation
    /// and symbols, then split on Unicode whitespace. No number expansion,
    /// stemming, filler removal, or accent folding. "don't" -> "dont".
    public static func words(_ text: String) -> [String] {
        let lower = text.precomposedStringWithCanonicalMapping.lowercased(with: Locale(identifier: "en_US_POSIX"))
        let removed = CharacterSet.punctuationCharacters.union(.symbols)
        let stripped = String(String.UnicodeScalarView(lower.unicodeScalars.filter { !removed.contains($0) }))
        return stripped.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// Word-level Levenshtein with unit edit costs and O(hypothesis words) memory.
    /// Equal-cost alignments prefer substitution, then deletion, then insertion.
    public static func score(reference: String, hypothesis: String) -> WordErrorScore {
        let reference = words(reference)
        let hypothesis = words(hypothesis)
        struct Edits {
            var substitutions = 0
            var deletions = 0
            var insertions = 0
            var total: Int { substitutions + deletions + insertions }
        }
        var previous = (0...hypothesis.count).map { Edits(insertions: $0) }
        for (i, referenceWord) in reference.enumerated() {
            var current = [Edits(deletions: i + 1)]
            for (j, hypothesisWord) in hypothesis.enumerated() {
                if referenceWord == hypothesisWord {
                    current.append(previous[j])
                } else {
                    var best = previous[j]
                    best.substitutions += 1
                    var deletion = previous[j + 1]
                    deletion.deletions += 1
                    if deletion.total < best.total { best = deletion }
                    var insertion = current[j]
                    insertion.insertions += 1
                    if insertion.total < best.total { best = insertion }
                    current.append(best)
                }
            }
            previous = current
        }
        let edits = previous[hypothesis.count]
        return WordErrorScore(referenceWordCount: reference.count, hypothesisWordCount: hypothesis.count,
                              substitutions: edits.substitutions, deletions: edits.deletions,
                              insertions: edits.insertions)
    }
}
