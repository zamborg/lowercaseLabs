import XCTest
@testable import blackhole

final class SSEParserTests: XCTestCase {
    func testParserBuildsMessageFromLines() {
        var parser = SSEParser()

        XCTAssertNil(parser.ingest(line: "event: final_result"))
        XCTAssertNil(parser.ingest(line: "data: {\"ok\":true}"))
        let message = parser.ingest(line: "")

        XCTAssertEqual(message, SSEMessage(event: "final_result", data: "{\"ok\":true}"))
    }
}

