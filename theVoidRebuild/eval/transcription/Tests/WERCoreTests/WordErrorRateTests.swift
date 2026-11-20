import Testing
@testable import WERCore

@Test func normalizesCasePunctuationAndWhitespace() {
    #expect(WordErrorRate.words("  HELLO,\nworld! Don't stop—now. $42 ") == ["hello", "world", "dont", "stopnow", "42"])
    #expect(WordErrorRate.score(reference: "Hello, WORLD!", hypothesis: "hello world").wer == 0)
}

@Test func preservesUnicodeAccentsAndNormalizesComposition() {
    #expect(WordErrorRate.words("CAFÉ cafe\u{301} naïve") == ["café", "café", "naïve"])
    #expect(WordErrorRate.words("I’M I’m") == ["im", "im"])
}

@Test func countsEachEditKind() {
    let substitution = WordErrorRate.score(reference: "a red car", hypothesis: "a blue car")
    #expect(substitution.substitutions == 1 && substitution.errorCount == 1)
    #expect(substitution.wer == 1.0 / 3.0)
    let deletion = WordErrorRate.score(reference: "a red car", hypothesis: "a car")
    #expect(deletion.deletions == 1 && deletion.errorCount == 1)
    let insertion = WordErrorRate.score(reference: "a car", hypothesis: "a red car")
    #expect(insertion.insertions == 1 && insertion.errorCount == 1)
}

@Test func alignsMixedEdits() {
    let score = WordErrorRate.score(reference: "a b c d e f", hypothesis: "a x c e f z")
    #expect(score.substitutions == 1)
    #expect(score.deletions == 1)
    #expect(score.insertions == 1)
    #expect(score.wer == 0.5)
}

@Test func handlesRepeatedWords() {
    let score = WordErrorRate.score(reference: "go go home", hypothesis: "go home home")
    #expect(score.errorCount == 1)
    #expect(score.substitutions == 1)
}

@Test func tiedAlignmentsPreferSubstitutions() {
    let score = WordErrorRate.score(reference: "a b", hypothesis: "b a")
    #expect(score.errorCount == 2)
    #expect(score.substitutions == 2)
}

@Test func emptyHypothesisIsAllDeletions() {
    let score = WordErrorRate.score(reference: "one two", hypothesis: "")
    #expect(score.deletions == 2)
    #expect(score.wer == 1)
}

@Test func emptyReferenceHasUndefinedWER() {
    let score = WordErrorRate.score(reference: "!!!", hypothesis: "one two")
    #expect(score.insertions == 2)
    #expect(score.wer == nil)
    #expect(WordErrorRate.score(reference: "", hypothesis: "").wer == nil)
}

@Test func werCanExceedOne() {
    #expect(WordErrorRate.score(reference: "one", hypothesis: "two three four").wer == 3)
}

@Test func corpusScoreWeightsByReferenceWordsAndIncludesSilenceInsertions() {
    let scores = [
        WordErrorRate.score(reference: "one", hypothesis: "two"),
        WordErrorRate.score(reference: "a b c d e f g h i", hypothesis: "a b c d e f g h i"),
        WordErrorRate.score(reference: "", hypothesis: "extra"),
    ]
    let aggregate = WordErrorScore.aggregate(scores)
    #expect(aggregate.referenceWordCount == 10)
    #expect(aggregate.errorCount == 2)
    #expect(aggregate.wer == 0.2)
    #expect(WordErrorScore.aggregate([]).wer == nil)
}
