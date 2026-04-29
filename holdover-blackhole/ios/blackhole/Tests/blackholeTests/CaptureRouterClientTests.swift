import XCTest
@testable import blackhole

final class CaptureRouterClientTests: XCTestCase {
    func testHighConfidenceTodoAutoCreates() throws {
        let client = DefaultCaptureRouterClient(autoRouteThreshold: 0.80)
        let result = TranscriptionFinalResult(
            sessionId: "session",
            transcript: "remember to file taxes tomorrow",
            route: .todo,
            routeConfidence: 0.91,
            routePayload: [
                "title": .string("File taxes"),
                "descriptionMarkdown": .string("remember to file taxes tomorrow"),
                "priority": .string("normal"),
                "deadlineAt": .null,
                "tags": .array([]),
            ],
            captureId: "capture"
        )

        let action = client.action(for: result)
        switch action {
        case .autoCreateTodo(let request):
            XCTAssertEqual(request.title, "File taxes")
            XCTAssertEqual(request.sourceCaptureId, "capture")
        default:
            XCTFail("Expected todo auto create")
        }
    }

    func testLowConfidenceNoteRequiresConfirmation() {
        let client = DefaultCaptureRouterClient(autoRouteThreshold: 0.80)
        let result = TranscriptionFinalResult(
            sessionId: "session",
            transcript: "product direction",
            route: .note,
            routeConfidence: 0.42,
            routePayload: ["title": .string("Product direction"), "bodyMarkdown": .string("product direction"), "tags": .array([])],
            captureId: nil
        )

        let action = client.action(for: result)
        if case .requireConfirmation = action {
            XCTAssertTrue(true)
        } else {
            XCTFail("Expected confirmation")
        }
    }
}
