import XCTest
import CoreGraphics
@testable import VeilCore

final class MatcherTests: XCTestCase {
    func matcher() throws -> Matcher { try Matcher(secrets: [Secret(label: "demo", value: "N8v2K6p4R9t3W5x7")]) }

    func testThresholdAndInterruptions() throws {
        let m = try matcher()
        for input in ["ordinary", "p4R9t", "p4R9 t3", "p4R9\nt3", "P4R9T3"] {
            XCTAssertTrue(m.matches(in: input).isEmpty, input)
        }
        XCTAssertEqual(m.matches(in: "p4R9t3").first?.range, NSRange(location: 0, length: 6))
    }

    func testOverlappingFragmentsMergeAndKeepUTF16Offsets() throws {
        let text = "🔒 key: N8v2K6p4R9t3W5x7 end"
        let expected = (text as NSString).range(of: "N8v2K6p4R9t3W5x7")
        let results = try matcher().matches(in: text)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.range, expected)
        XCTAssertEqual(results.first?.labels, ["demo"])
    }

    func testSharedFragmentsIdentifyAllCandidates() throws {
        let m = try Matcher(secrets: [Secret(label: "A", value: "abcdef123"), Secret(label: "B", value: "abcdef987")])
        XCTAssertEqual(m.matches(in: "abcdef").first?.labels, ["A", "B"])
    }

    func testSeparatedOccurrencesAndVisibleRangeOffsets() throws {
        let result = try matcher().matches(in: "p4R9t3 / p4R9t3", offset: 101)
        XCTAssertEqual(result.map(\.range), [NSRange(location: 101, length: 6), NSRange(location: 110, length: 6)])
    }

    func testInvalidConfiguration() {
        for value in ["", "short", "with space", "αβγδεζη"] {
            XCTAssertThrowsError(try Matcher(secrets: [Secret(label: "A", value: value)]))
        }
        XCTAssertThrowsError(try Matcher(secrets: [], threshold: 0))
        XCTAssertThrowsError(try Matcher(secrets: [], threshold: 1025))
        XCTAssertThrowsError(try Matcher(secrets: [Secret(label: "", value: "abcdef")]))
    }

    func testConfigurableThresholds() throws {
        let secrets = [Secret(label: "demo", value: "N8v2K6p4R9t3W5x7")]
        for threshold in [1, 4, 6, 8, 16] {
            let matcher = try Matcher(secrets: secrets, threshold: threshold)
            let value = secrets[0].value
            XCTAssertTrue(matcher.matches(in: String(value.prefix(threshold - 1))).isEmpty)
            XCTAssertEqual(matcher.matches(in: String(value.prefix(threshold))).first?.range, NSRange(location: 0, length: threshold))
            XCTAssertEqual(matcher.matches(in: "🔒 " + value).first?.range, NSRange(location: 3, length: 16))
        }
    }

    func testThresholdLongerThanSecretIsRejected() {
        XCTAssertThrowsError(try Matcher(secrets: [Secret(label: "short key", value: "abcdef")], threshold: 7))
    }

    func testAppScopeSelectionAndSelfExclusion() {
        let own = "labs.lowercase.veil"
        let selected = AppScope(allApps: false, bundleIDs: ["com.apple.TextEdit", "com.googlecode.iterm2"])
        XCTAssertTrue(selected.allows("com.apple.TextEdit", excluding: own))
        XCTAssertTrue(selected.allows("com.googlecode.iterm2", excluding: own))
        XCTAssertFalse(selected.allows("com.apple.Terminal", excluding: own))
        XCTAssertFalse(selected.allows(nil, excluding: own))
        XCTAssertFalse(AppScope(allApps: true, bundleIDs: []).allows(own, excluding: own))
        XCTAssertTrue(AppScope(allApps: true, bundleIDs: []).allows("any.app", excluding: own))
        XCTAssertFalse(AppScope(allApps: false, bundleIDs: []).allows("any.app", excluding: own))
    }

    func testDisplayCoordinateConversion() {
        XCTAssertEqual(Geometry.appKitRect(CGRect(x: -1000, y: 150, width: 100, height: 30), primaryHeight: 900), CGRect(x: -1000, y: 720, width: 100, height: 30))
    }
}
