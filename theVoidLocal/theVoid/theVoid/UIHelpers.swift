import SwiftUI
import UIKit

// MARK: - Helpers

extension Color {
    init(hex: String) {
        var sanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        sanitized = sanitized.replacingOccurrences(of: "#", with: "")

        var value: UInt64 = 0
        Scanner(string: sanitized).scanHexInt64(&value)

        let r, g, b: Double
        if sanitized.count == 6 {
            r = Double((value & 0xFF0000) >> 16) / 255.0
            g = Double((value & 0x00FF00) >> 8) / 255.0
            b = Double(value & 0x0000FF) / 255.0
        } else {
            r = 0.5
            g = 0.5
            b = 0.5
        }

        self.init(red: r, green: g, blue: b)
    }
}

extension DateFormatter {
    static let hhmm: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        return formatter
    }()

    static let localDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "h:mm a"
        return formatter
    }()
}

extension UIApplication {
    static func openHealthAccessManagement() {
        let app = UIApplication.shared
        guard let healthURL = URL(string: "x-apple-health://") else {
            if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                app.open(settingsURL)
            }
            return
        }

        app.open(healthURL, options: [:]) { opened in
            guard !opened else { return }
            if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                app.open(settingsURL)
            }
        }
    }
}
