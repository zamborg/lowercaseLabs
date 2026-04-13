import Foundation
import Observation

@MainActor
@Observable
final class TodosViewModel {
    enum Filter: String, CaseIterable, Identifiable {
        case open = "Open"
        case all = "All"
        case done = "Done"

        var id: String { rawValue }

        var apiValue: String? {
            switch self {
            case .open:
                return "open"
            case .all:
                return "all"
            case .done:
                return "completed"
            }
        }
    }

    var todos: [TodoRecord] = []
    var filter: Filter = .open
    var isLoading = false
    var errorMessage: String?

    private let todosRepository: TodosRepository
    private let navigator: AppNavigator

    init(todosRepository: TodosRepository, navigator: AppNavigator) {
        self.todosRepository = todosRepository
        self.navigator = navigator
    }

    func refresh() async {
        guard !isLoading else { return }
        isLoading = true
        defer { isLoading = false }
        do {
            let page = try await todosRepository.listTodos(status: filter.apiValue)
            todos = page.items
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func save(todo: TodoRecord) async {
        do {
            let updated = try await todosRepository.updateTodo(
                id: todo.id,
                request: TodoUpdateRequest(
                    title: todo.title,
                    descriptionMarkdown: todo.descriptionMarkdown,
                    priority: todo.priority,
                    deadlineAt: todo.deadlineAt,
                    tags: todo.tags,
                    status: todo.status
                )
            )
            if let index = todos.firstIndex(where: { $0.id == updated.id }) {
                todos[index] = updated
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func triggerVoiceAdd() {
        navigator.startVoiceTodo()
    }
}

