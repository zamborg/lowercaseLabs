import AppKit
import VeilCore

final class MaskPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

final class MaskView: NSView {
    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill()
        bounds.fill()
        let font = NSFont.monospacedSystemFont(ofSize: min(18, max(10, bounds.height - 5)), weight: .bold)
        let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
        let stars = String(repeating: "*", count: max(1, min(80, Int(bounds.width / max(6, font.pointSize * 0.6)))))
        let size = (stars as NSString).size(withAttributes: attributes)
        (stars as NSString).draw(at: NSPoint(x: 1, y: max(0, (bounds.height - size.height) / 2)), withAttributes: attributes)
    }
}

final class Overlay {
    private var panels: [MaskPanel] = []
    func hide() { panels.forEach { $0.orderOut(nil) } }
    func show(_ rectangles: [CGRect]) {
        guard let primaryHeight = NSScreen.screens.first?.frame.height else { hide(); return }
        for (index, rect) in rectangles.prefix(128).enumerated() {
            if index == panels.count {
                let panel = MaskPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
                panel.backgroundColor = .black
                panel.isOpaque = true
                panel.hasShadow = false
                panel.ignoresMouseEvents = true
                panel.hidesOnDeactivate = false
                panel.level = .floating
                panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
                let view = MaskView()
                view.setAccessibilityElement(false)
                panel.contentView = view
                panels.append(panel)
            }
            let frame = Geometry.appKitRect(rect.insetBy(dx: -2, dy: -1), primaryHeight: primaryHeight).integral
            panels[index].setFrame(frame, display: true)
            panels[index].orderFrontRegardless()
        }
        for panel in panels.dropFirst(min(128, rectangles.count)) { panel.orderOut(nil) }
    }
}
