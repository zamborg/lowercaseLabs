import AppKit
import ApplicationServices
import VeilCore

struct ScanResult {
    var rectangles: [CGRect] = []
    var labels: Set<String> = []
    var inspected = 0
    var unsupported = 0
    var broadMasks = 0
    var limited = false
    var message: String?
}

enum AXRead {
    static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
    }
    static func parameter(_ element: AXUIElement, _ name: String, _ value: CFTypeRef) -> CFTypeRef? {
        var result: CFTypeRef?
        return AXUIElementCopyParameterizedAttributeValue(element, name as CFString, value, &result) == .success ? result : nil
    }
    static func element(_ value: CFTypeRef?) -> AXUIElement? {
        guard let value, CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return unsafeBitCast(value, to: AXUIElement.self)
    }
    static func unpack<T>(_ value: CFTypeRef?, _ type: AXValueType, into result: inout T) -> Bool {
        guard let value, CFGetTypeID(value) == AXValueGetTypeID() else { return false }
        let ax = unsafeBitCast(value, to: AXValue.self)
        guard AXValueGetType(ax) == type else { return false }
        return withUnsafeMutablePointer(to: &result) { AXValueGetValue(ax, type, $0) }
    }
    static func frame(_ element: AXUIElement) -> CGRect? {
        var point = CGPoint.zero; var size = CGSize.zero
        guard unpack(attribute(element, kAXPositionAttribute), .cgPoint, into: &point),
              unpack(attribute(element, kAXSizeAttribute), .cgSize, into: &size), size.width > 0, size.height > 0 else { return nil }
        return CGRect(origin: point, size: size)
    }
    static func bounds(_ element: AXUIElement, range: NSRange) -> CGRect? {
        var r = CFRange(location: range.location, length: range.length)
        guard let value = AXValueCreate(.cfRange, &r) else { return nil }
        var rect = CGRect.zero
        guard unpack(parameter(element, kAXBoundsForRangeParameterizedAttribute, value), .cgRect, into: &rect),
              !rect.isNull, !rect.isInfinite, rect.width > 0, rect.height > 0 else { return nil }
        return rect
    }
}

enum AccessibilityScanner {
    static func scan(pid: pid_t, matcher: Matcher) -> ScanResult {
        var result = ScanResult()
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.05)
        guard let window = AXRead.element(AXRead.attribute(app, kAXFocusedWindowAttribute)),
              let windowFrame = AXRead.frame(window) else {
            result.message = "No accessible front window."
            return result
        }
        let deadline = Date().addingTimeInterval(0.25)
        var queue = [window]; var index = 0
        while index < queue.count && index < 800 && Date() < deadline {
            let element = queue[index]; index += 1
            let role = AXRead.attribute(element, kAXRoleAttribute) as? String ?? ""
            if [kAXTextAreaRole, kAXTextFieldRole, kAXStaticTextRole].contains(role) {
                inspect(element, role: role, matcher: matcher, clip: windowFrame, deadline: deadline, result: &result)
                // A text area's static-text children often repeat its entire value.
                if role == kAXTextAreaRole { continue }
            }
            var children: CFArray?
            if AXUIElementCopyAttributeValues(element, kAXChildrenAttribute as CFString, 0, 800 - min(queue.count, 800), &children) == .success,
               let children = children as? [AXUIElement] { queue.append(contentsOf: children) }
            if result.rectangles.count >= 128 { result.limited = true; break }
        }
        result.limited = result.limited || index < queue.count
        return result
    }

    private static func inspect(_ element: AXUIElement, role: String, matcher: Matcher, clip: CGRect, deadline: Date, result: inout ScanResult) {
        guard let elementFrame = AXRead.frame(element) else { result.unsupported += 1; return }
        let region = clip.intersection(elementFrame)
        guard !region.isNull, region.width > 0, region.height > 0 else { return }
        var visible = CFRange(location: 0, length: 0)
        let hasVisible = AXRead.unpack(AXRead.attribute(element, kAXVisibleCharacterRangeAttribute), .cfRange, into: &visible)
        if hasVisible && visible.length == 0 { return }
        var text: String?
        var offset = 0
        if hasVisible, visible.location >= 0, visible.length > 0, visible.length <= 100_000 {
            offset = visible.location
            if let range = AXValueCreate(.cfRange, &visible) {
                text = AXRead.parameter(element, kAXStringForRangeParameterizedAttribute, range) as? String
            }
            if text == nil, let value = AXRead.attribute(element, kAXValueAttribute) as? String {
                let ns = value as NSString
                if visible.location <= ns.length && visible.length <= ns.length - visible.location {
                    text = ns.substring(with: NSRange(location: visible.location, length: visible.length))
                }
            }
        } else if role != kAXTextAreaRole {
            text = (AXRead.attribute(element, kAXValueAttribute) as? String) ?? (AXRead.attribute(element, kAXTitleAttribute) as? String)
        }
        guard let text, text.utf16.count <= 100_000 else { result.unsupported += 1; return }
        result.inspected += 1
        let matches = matcher.matches(in: text, offset: offset)
        guard !matches.isEmpty else { return }
        for match in matches {
            result.labels.formUnion(match.labels)
            // Query each glyph so wrapping never yields a huge cross-line rectangle.
            // If geometry is missing, conceal the entire known text element.
            var pieces: [CGRect] = []
            var reliable = match.range.length <= 1024
            if reliable {
                for position in match.range.location..<NSMaxRange(match.range) {
                    guard Date() < deadline, let rawBounds = AXRead.bounds(element, range: NSRange(location: position, length: 1)) else {
                        reliable = false; break
                    }
                    let bounds = rawBounds.intersection(region)
                    if !bounds.isNull && bounds.width > 0 && bounds.height > 0 {
                        if let last = pieces.last, abs(last.minY - bounds.minY) < 2, abs(last.height - bounds.height) < 2,
                           bounds.minX <= last.maxX + 3, bounds.maxX >= last.minX - 3 {
                            pieces[pieces.count - 1] = last.union(bounds)
                        } else { pieces.append(bounds) }
                    }
                }
            }
            if reliable { result.rectangles.append(contentsOf: pieces) }
            else { result.broadMasks += 1; result.rectangles.append(region) }
        }
    }
}
