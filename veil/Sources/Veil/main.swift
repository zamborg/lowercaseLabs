import AppKit
import ApplicationServices
import VeilCore

final class AppDelegate: NSObject, NSApplicationDelegate, NSTextFieldDelegate {
    private var window: NSWindow!
    private var statusItem: NSStatusItem!
    private let status = NSTextField(wrappingLabelWithString: "Paused. Configure secrets or use the synthetic demo.")
    private let appPicker = NSPopUpButton(frame: .zero, pullsDown: true)
    private let minimumField = NSTextField(string: "6")
    private let minimumStepper = NSStepper()
    private let secrets = NSTextView()
    private let toggle = NSButton(title: "Start masking", target: nil, action: nil)
    private let overlay = Overlay()
    private let worker = DispatchQueue(label: "Veil.accessibility", qos: .userInitiated)
    private var timer: Timer?
    private var matcher: Matcher?
    private var scope = AppScope(allApps: false, bundleIDs: ["com.googlecode.iterm2"])
    private var minimumLength = 6
    private var configuredSecrets: [Secret] = []
    private var appChoices: [AppChoice] = []
    private var enabled = false
    private var busy = false
    private var generation = 0
    private var lastPID: pid_t?
    private var monitoringActivity: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        let defaults = UserDefaults.standard
        scope.allApps = defaults.bool(forKey: "allApps")
        scope.bundleIDs = Set(defaults.stringArray(forKey: "selectedApps") ?? ["com.googlecode.iterm2"])
        let savedMinimum = defaults.object(forKey: "minimumLength") as? Int ?? 6
        minimumLength = (1...1024).contains(savedMinimum) ? savedMinimum : 6
        buildMenu()
        buildWindow()
        let polling = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(polling, forMode: .common)
        timer = polling
        NSWorkspace.shared.notificationCenter.addObserver(self, selector: #selector(activeAppChanged), name: NSWorkspace.didActivateApplicationNotification, object: nil)
        showSettings()
    }

    private func buildMenu() {
        let menu = NSMenu()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Settings…", action: #selector(showSettings), keyEquivalent: ",").target = self
        appMenu.addItem(withTitle: "Pause", action: #selector(pause), keyEquivalent: "p").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Veil", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let parent = NSMenuItem(); parent.submenu = appMenu; menu.addItem(parent)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let editParent = NSMenuItem(); editParent.submenu = edit; menu.addItem(editParent)
        NSApp.mainMenu = menu
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "Veil ○"
        let tray = NSMenu()
        tray.addItem(withTitle: "Settings…", action: #selector(showSettings), keyEquivalent: "").target = self
        tray.addItem(withTitle: "Pause", action: #selector(pause), keyEquivalent: "").target = self
        tray.addItem(withTitle: "Quit Veil", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "")
        statusItem.menu = tray
    }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 700, height: 690), styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "Veil — visual masking prototype"
        window.isReleasedWhenClosed = false
        window.center()
        let stack = NSStackView()
        stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        window.contentView!.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor, constant: -24),
            stack.topAnchor.constraint(equalTo: window.contentView!.topAnchor, constant: 22)
        ])
        let heading = NSTextField(labelWithString: "Hide known secrets as asterisks")
        heading.font = .systemFont(ofSize: 22, weight: .semibold)
        stack.addArrangedSubview(heading)
        let intro = NSTextField(wrappingLabelWithString: "Veil covers matching text in the active window of your chosen apps. Choose All apps to follow the active app automatically. Original text is preserved.")
        stack.addArrangedSubview(intro)
        intro.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        stack.addArrangedSubview(NSTextField(labelWithString: "Applications — check one or more"))
        appPicker.setAccessibilityIdentifier("applicationPicker")
        stack.addArrangedSubview(appPicker)
        appPicker.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        refreshApps()
        let minimumRow = NSStackView(); minimumRow.orientation = .horizontal; minimumRow.spacing = 10
        minimumRow.addArrangedSubview(NSTextField(labelWithString: "Minimum matching substring"))
        minimumField.integerValue = minimumLength
        minimumField.delegate = self
        minimumField.target = self; minimumField.action = #selector(minimumChanged)
        minimumField.setAccessibilityIdentifier("minimumMatchLength")
        minimumField.widthAnchor.constraint(equalToConstant: 70).isActive = true
        minimumRow.addArrangedSubview(minimumField)
        minimumStepper.minValue = 1; minimumStepper.maxValue = 1024; minimumStepper.increment = 1
        minimumStepper.integerValue = minimumLength
        minimumStepper.target = self; minimumStepper.action = #selector(stepMinimum)
        minimumStepper.setAccessibilityIdentifier("minimumMatchStepper")
        minimumRow.addArrangedSubview(minimumStepper)
        minimumRow.addArrangedSubview(NSTextField(labelWithString: "consecutive characters (1–1024)"))
        stack.addArrangedSubview(minimumRow)
        let scopeNote = NSTextField(wrappingLabelWithString: "App selection and match length are saved and apply immediately. All apps scans the active window; inactive windows are outside this version’s scope. Smaller lengths produce more matches.")
        scopeNote.font = .systemFont(ofSize: 12); scopeNote.textColor = .secondaryLabelColor
        stack.addArrangedSubview(scopeNote)
        scopeNote.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        stack.addArrangedSubview(NSTextField(labelWithString: "Secrets — one label=secret per line (kept in memory for this session)"))
        secrets.isRichText = false; secrets.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        secrets.isAutomaticQuoteSubstitutionEnabled = false
        secrets.isAutomaticDashSubstitutionEnabled = false
        secrets.isAutomaticSpellingCorrectionEnabled = false
        secrets.setAccessibilityIdentifier("secretConfiguration")
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true; scroll.borderType = .bezelBorder; scroll.documentView = secrets
        secrets.autoresizingMask = [.width]; secrets.isVerticallyResizable = true
        secrets.textContainer?.widthTracksTextView = true
        stack.addArrangedSubview(scroll)
        scroll.heightAnchor.constraint(equalToConstant: 130).isActive = true
        scroll.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        let buttons = NSStackView(); buttons.orientation = .horizontal; buttons.spacing = 10
        toggle.target = self; toggle.action = #selector(startOrPause)
        buttons.addArrangedSubview(toggle)
        let demo = NSButton(title: "Load TextEdit demo", target: self, action: #selector(loadDemo))
        buttons.addArrangedSubview(demo)
        let permission = NSButton(title: "Accessibility access…", target: self, action: #selector(requestAccess))
        buttons.addArrangedSubview(permission)
        stack.addArrangedSubview(buttons)
        status.font = .systemFont(ofSize: 12, weight: .medium)
        status.setAccessibilityIdentifier("monitorStatus")
        stack.addArrangedSubview(status)
        status.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        let limitation = NSTextField(wrappingLabelWithString: "Best-effort overlay: text may appear before it is covered. Window-only capture may omit the masks. Unsupported text is reported as incomplete coverage. Use synthetic data to test first.")
        limitation.font = .systemFont(ofSize: 12); limitation.textColor = .secondaryLabelColor
        stack.addArrangedSubview(limitation)
        limitation.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
    }

    @objc private func showSettings() { overlay.hide(); window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true) }
    @objc private func refreshApps() {
        appChoices = AppCatalog.choices(including: scope.bundleIDs)
        rebuildPicker()
    }
    private func rebuildPicker() {
        let menu = NSMenu(); menu.autoenablesItems = false
        let names = appChoices.filter { scope.bundleIDs.contains($0.bundleID) }.map(\.name)
        let summary = scope.allApps ? "All apps — active window" : names.isEmpty ? "Choose apps…" : names.joined(separator: ", ")
        menu.addItem(withTitle: summary, action: nil, keyEquivalent: "")
        let all = menu.addItem(withTitle: "All apps — active window", action: #selector(selectAllApps), keyEquivalent: "")
        all.target = self; all.state = scope.allApps ? .on : .off
        menu.addItem(.separator())
        for choice in appChoices {
            let item = menu.addItem(withTitle: choice.name, action: #selector(selectApp(_:)), keyEquivalent: "")
            item.target = self; item.representedObject = choice.bundleID
            item.toolTip = choice.bundleID
            item.state = !scope.allApps && scope.bundleIDs.contains(choice.bundleID) ? .on : .off
        }
        menu.addItem(.separator())
        menu.addItem(withTitle: "Refresh app list", action: #selector(refreshApps), keyEquivalent: "").target = self
        appPicker.menu = menu
    }
    @objc private func selectAllApps() {
        scope.allApps.toggle()
        selectionChanged()
    }
    @objc private func selectApp(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        if scope.allApps {
            scope.allApps = false
            scope.bundleIDs = [id]
            selectionChanged()
            return
        }
        scope.allApps = false
        if scope.bundleIDs.contains(id) { scope.bundleIDs.remove(id) } else { scope.bundleIDs.insert(id) }
        selectionChanged()
    }
    private func selectionChanged() {
        generation += 1; overlay.hide(); lastPID = nil
        UserDefaults.standard.set(scope.allApps, forKey: "allApps")
        UserDefaults.standard.set(scope.bundleIDs.sorted(), forKey: "selectedApps")
        rebuildPicker()
        status.stringValue = "Application selection updated. Only the active window of a selected app is scanned."
    }
    @objc private func stepMinimum() {
        minimumField.integerValue = minimumStepper.integerValue
        minimumChanged()
    }
    @objc private func minimumChanged() {
        do {
            guard let value = Int(minimumField.stringValue), (1...1024).contains(value) else {
                throw ConfigurationError("Enter a whole-number match length from 1 to 1024.")
            }
            guard value != minimumLength else { return }
            let updated = enabled ? try Matcher(secrets: configuredSecrets, threshold: value) : nil
            minimumLength = value; minimumStepper.integerValue = value
            UserDefaults.standard.set(value, forKey: "minimumLength")
            if let updated { generation += 1; overlay.hide(); matcher = updated }
            status.stringValue = "Minimum match length: \(value) consecutive characters."
        } catch {
            minimumField.integerValue = minimumLength; minimumStepper.integerValue = minimumLength
            status.stringValue = error.localizedDescription
        }
    }
    func controlTextDidEndEditing(_ notification: Notification) {
        if notification.object as? NSTextField === minimumField { minimumChanged() }
    }
    @objc private func loadDemo() {
        pause()
        scope = AppScope(allApps: false, bundleIDs: ["com.apple.TextEdit"])
        selectionChanged()
        minimumField.integerValue = 6; minimumChanged()
        secrets.string = "Synthetic key A=N8v2K6p4R9t3W5x7"
        status.stringValue = "Demo loaded. Start masking, then focus TextEdit with the synthetic key visible."
    }
    @objc private func requestAccess() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        status.stringValue = "Enable Veil in System Settings → Privacy & Security → Accessibility. Then start masking."
    }
    @objc private func pause() {
        generation += 1; enabled = false; overlay.hide(); lastPID = nil
        if let activity = monitoringActivity { ProcessInfo.processInfo.endActivity(activity); monitoringActivity = nil }
        toggle.title = "Start masking"; statusItem.button?.title = "Veil ○"
        status.stringValue = "Paused. Masks removed."
    }
    @objc private func startOrPause() {
        if enabled { pause(); return }
        do {
            let parsed = try secrets.string.split(whereSeparator: \.isNewline).map { line -> Secret in
                guard let separator = line.firstIndex(of: "=") else { throw ConfigurationError("Use label=secret on each line.") }
                return Secret(label: String(line[..<separator]).trimmingCharacters(in: .whitespaces), value: String(line[line.index(after: separator)...]))
            }
            guard !parsed.isEmpty else { throw ConfigurationError("Add at least one secret or load the synthetic demo.") }
            guard scope.allApps || !scope.bundleIDs.isEmpty else { throw ConfigurationError("Select at least one application, or choose All apps.") }
            guard let value = Int(minimumField.stringValue), (1...1024).contains(value) else { throw ConfigurationError("Enter a whole-number match length from 1 to 1024.") }
            let compiled = try Matcher(secrets: parsed, threshold: value)
            guard AXIsProcessTrusted() else {
                let front = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unavailable"
                status.stringValue = "Accessibility access is required. Use the access button, enable Veil, then retry. Front app: \(front)."
                return
            }
            minimumLength = value; minimumStepper.integerValue = value
            UserDefaults.standard.set(value, forKey: "minimumLength")
            matcher = compiled; configuredSecrets = parsed; generation += 1; enabled = true
            monitoringActivity = ProcessInfo.processInfo.beginActivity(options: .userInitiatedAllowingIdleSystemSleep, reason: "Track visual secret masks while another app is active")
            toggle.title = "Pause masking"; statusItem.button?.title = "Veil ●"
            status.stringValue = "Ready. Focus any selected app. Secret edits apply after pausing and restarting."
        } catch { status.stringValue = error.localizedDescription }
    }
    @objc private func activeAppChanged() {
        // Remove stale overlays immediately when a different application takes focus.
        generation += 1; overlay.hide(); lastPID = nil
    }
    private func tick() {
        guard enabled else { return }
        guard AXIsProcessTrusted() else { pause(); status.stringValue = "Accessibility access was revoked. Monitoring paused."; return }
        guard let app = NSWorkspace.shared.frontmostApplication,
              scope.allows(app.bundleIdentifier, excluding: Bundle.main.bundleIdentifier ?? "labs.lowercase.veil"), let matcher else {
            overlay.hide()
            let front = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unavailable"
            status.stringValue = "Waiting for a selected app’s active window. Front app: \(front). Secret edits apply after pausing and restarting."
            return
        }
        guard !busy else { return }
        busy = true
        let epoch = generation; let pid = app.processIdentifier
        if lastPID != pid { overlay.hide(); lastPID = pid }
        worker.async {
            let result = AccessibilityScanner.scan(pid: pid, matcher: matcher)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.busy = false
                guard self.enabled, self.generation == epoch, NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else { return }
                self.overlay.show(result.rectangles)
                let coverage = result.limited || result.unsupported > 0 ? "Incomplete coverage" : "AX scan completed"
                let matches = result.labels.isEmpty ? "No matches" : "Matched labels: " + result.labels.sorted().joined(separator: ", ")
                self.status.stringValue = result.message ?? "\(matches). \(result.rectangles.count) masks; \(result.broadMasks) whole-element fallbacks. \(coverage); \(result.inspected) text elements checked, \(result.unsupported) unsupported."
                self.statusItem.button?.title = result.rectangles.isEmpty ? "Veil ●" : "Veil *\(result.rectangles.count)"
            }
        }
    }
    func applicationWillTerminate(_ notification: Notification) { pause(); timer?.invalidate() }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
