import Foundation

enum CaptureAutoAction: Sendable {
    case autoCreateNote(NoteCreateRequest)
    case autoCreateTodo(TodoCreateRequest)
    case openSearch(String)
    case requireConfirmation
}

protocol CaptureRouterClient {
    func action(for result: TranscriptionFinalResult) -> CaptureAutoAction
}

struct DefaultCaptureRouterClient: CaptureRouterClient {
    let autoRouteThreshold: Double

    init(autoRouteThreshold: Double = 0.80) {
        self.autoRouteThreshold = autoRouteThreshold
    }

    func action(for result: TranscriptionFinalResult) -> CaptureAutoAction {
        switch result.route {
        case .search:
            let payload = (try? result.routePayload.decodePayload(SearchDraftPayload.self)) ?? SearchDraftPayload(queryText: result.transcript)
            return .openSearch(payload.queryText)
        case .note:
            guard result.routeConfidence >= autoRouteThreshold,
                  let payload = try? result.routePayload.decodePayload(NoteDraftPayload.self) else {
                return .requireConfirmation
            }
            return .autoCreateNote(
                NoteCreateRequest(
                    title: payload.title,
                    bodyMarkdown: payload.bodyMarkdown,
                    tags: payload.tags,
                    sourceCaptureId: result.captureId
                )
            )
        case .todo:
            guard result.routeConfidence >= autoRouteThreshold,
                  let payload = try? result.routePayload.decodePayload(TodoDraftPayload.self) else {
                return .requireConfirmation
            }
            return .autoCreateTodo(
                TodoCreateRequest(
                    title: payload.title,
                    descriptionMarkdown: payload.descriptionMarkdown,
                    priority: payload.priority,
                    deadlineAt: payload.deadlineAt,
                    tags: payload.tags,
                    sourceCaptureId: result.captureId
                )
            )
        }
    }
}

