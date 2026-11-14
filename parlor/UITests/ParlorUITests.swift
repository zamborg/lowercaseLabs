import XCTest

final class ParlorUITests: XCTestCase {
    var app: XCUIApplication!
    var home: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false
        home = FileManager.default.temporaryDirectory.appendingPathComponent("parlor-ui-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        app = XCUIApplication()
        app.launchEnvironment["PARLOR_HOME"] = home.path
        app.launchEnvironment["PARLOR_DEMO"] = "1"
    }

    func testCreateChatPruneRelaunch() throws {
        app.launch()
        // Seeded agents appear in the sidebar.
        XCTAssertTrue(app.staticTexts["Mabel"].firstMatch.waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Pip"].firstMatch.exists)

        // Chat with Mabel.
        app.staticTexts["Mabel"].firstMatch.click()
        let composer = app.textFields["composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        composer.click()
        composer.typeText("hello there\n")
        XCTAssertTrue(app.staticTexts["(demo) You said: hello there"].waitForExistence(timeout: 10), "reply streams in")

        // Edit the agent: rename Pip → the file on disk follows.
        app.staticTexts["Pip"].firstMatch.click()
        app.radioButtons["Agent"].firstMatch.click()
        let name = app.textFields["agent-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.click()
        name.typeKey("a", modifierFlags: .command)
        name.typeText("Pippa")
        let file = home.appendingPathComponent("agents/pip.md")
        let deadline = Date().addingTimeInterval(5)
        var onDisk = ""
        while Date() < deadline {
            onDisk = (try? String(contentsOf: file, encoding: .utf8)) ?? ""
            if onDisk.contains("name: Pippa") { break }
            Thread.sleep(forTimeInterval: 0.2)
        }
        XCTAssertTrue(onDisk.contains("name: Pippa"), "editor saves to agents/pip.md, got: \(onDisk.prefix(80))")

        // Prune: back to Mabel, select the user message, delete it.
        app.staticTexts["Mabel"].firstMatch.click()
        app.radioButtons["Chat"].firstMatch.click()
        app.menuButtons["prune-menu"].firstMatch.click()
        app.menuItems["Select Messages…"].click()
        app.staticTexts["hello there"].firstMatch.click()
        app.buttons["delete-selected"].click()
        XCTAssertFalse(app.staticTexts["hello there"].exists, "message deleted")

        // Relaunch: chat and agent edits survive.
        app.terminate()
        app.launch()
        XCTAssertTrue(app.staticTexts["Pippa"].firstMatch.waitForExistence(timeout: 15))
    }

    func testRoomRoundRobin() throws {
        let dir = home.appendingPathComponent("chats/duo-ab12")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = #"{"version":1,"title":"Duo","participants":[{"agent":"mabel"},{"agent":"pip"},{"user":true}],"scheduler":{"kind":"roundRobin","maxTurns":2}}"#
        try json.write(to: dir.appendingPathComponent("conversation.json"), atomically: true, encoding: .utf8)
        app.launchEnvironment["PARLOR_ROUTE"] = "room:duo-ab12"
        app.launch()
        let composer = app.textFields["room-composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 15))
        composer.click()
        composer.typeText("hi both\n")
        let reply = app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).firstMatch
        if !reply.waitForExistence(timeout: 15) { print(app.debugDescription) }
        XCTAssertTrue(reply.exists)
        // Two agents answered: both names appear as speaker labels.
        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline, app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).count < 2 { Thread.sleep(forTimeInterval: 0.3) }
        XCTAssertGreaterThanOrEqual(app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).count, 2)
        let shot = XCUIScreen.main.screenshot()
        let att = XCTAttachment(screenshot: shot); att.lifetime = .keepAlways; add(att)
    }

    func testSystemServices() throws {
        let script = home.appendingPathComponent("fake.py")
        let py = """
        import sys, json
        def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        for line in sys.stdin:
            m = json.loads(line); i = m.get("id"); method = m.get("method")
            if method == "initialize": send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}})
            elif method == "tools/list": send({"jsonrpc":"2.0","id":i,"result":{"tools":[]}})
        """
        try py.write(to: script, atomically: true, encoding: .utf8)
        let mcp = #"{"mcpServers":{"fake":{"command":"python3","args":["\#(script.path)"]},"broken":{"command":"/no/such/command"}}}"#
        try mcp.write(to: home.appendingPathComponent("mcp.json"), atomically: true, encoding: .utf8)
        try #"{"providers":[{"id":"nowhere","baseURL":"http://127.0.0.1:9/v1","models":["m"]}]}"#.write(to: home.appendingPathComponent("settings.json"), atomically: true, encoding: .utf8)
        app.launchEnvironment["PARLOR_ROUTE"] = "system"
        app.launch()
        XCTAssertTrue(app.buttons["stop-mcp:fake"].waitForExistence(timeout: 20), "fake server running")
        XCTAssertTrue(app.buttons["start-mcp:broken"].waitForExistence(timeout: 5), "failed server can be started again")
        app.buttons["stop-mcp:fake"].click()
        XCTAssertTrue(app.buttons["start-mcp:fake"].waitForExistence(timeout: 10), "stopped from System")
        app.buttons["start-mcp:fake"].click()
        XCTAssertTrue(app.buttons["stop-mcp:fake"].waitForExistence(timeout: 20), "started again")
        XCTAssertTrue(app.staticTexts["Your data"].exists)
        // Providers: the configured one is listed and can be tested; an unreachable one says so.
        let test = app.buttons["test-provider-nowhere"]
        XCTAssertTrue(test.waitForExistence(timeout: 5), "provider listed with a Test button")
        test.click()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'Failed'")).firstMatch.waitForExistence(timeout: 10), "unreachable provider reports failure")
        XCTAssertTrue(app.staticTexts["Runtime API"].exists, "runtime API is a service")
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot()); shot.lifetime = .keepAlways; add(shot)
    }

    /// Scrolls the editor form (by wheeling over a heading that is on screen) until `element` can be clicked.
    private func scrollIntoView(_ element: XCUIElement) {
        guard let form = app.scrollViews.allElementsBoundByIndex.max(by: { $0.frame.height < $1.frame.height }) else { return }
        for _ in 0..<14 where !element.isHittable { form.scroll(byDeltaX: 0, deltaY: -250) }
    }

    /// Step 7 screens: Tools and History live in System; the editor has memory and per-agent limits.
    func testStep7Screens() throws {
        app.launchEnvironment["PARLOR_ROUTE"] = "tools"          // an alias of system
        app.launch()
        XCTAssertTrue(app.staticTexts["History"].waitForExistence(timeout: 15), "History section in System")
        XCTAssertTrue(app.buttons["add-server"].exists, "Tools panel moved into System")
        app.terminate()

        app.launchEnvironment["PARLOR_ROUTE"] = "agent:mabel"
        app.launchEnvironment["PARLOR_MODE"] = "edit"
        app.launch()
        let memory = app.textViews["agent-memory"]
        XCTAssertTrue(memory.waitForExistence(timeout: 15), "memory editor")
        scrollIntoView(memory)
        memory.click()
        memory.typeText("Likes oolong.")
        let file = home.appendingPathComponent("memory/mabel.md")
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline, !FileManager.default.fileExists(atPath: file.path) { Thread.sleep(forTimeInterval: 0.3) }
        XCTAssertEqual((try? String(contentsOf: file, encoding: .utf8))?.contains("Likes oolong."), true, "memory is saved to memory/<id>.md")
        let override = app.buttons["override-top_p"]
        XCTAssertTrue(override.waitForExistence(timeout: 5))
        scrollIntoView(override)
        override.click()
        XCTAssertTrue(app.buttons["inherit-top_p"].waitForExistence(timeout: 5), "override shows Inherit")
        app.buttons["inherit-top_p"].click()
        XCTAssertTrue(app.buttons["override-top_p"].waitForExistence(timeout: 5))
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot()); shot.lifetime = .keepAlways; add(shot)
    }

    /// A token made in System drives the running app over HTTP, and a module needs the user's approval.
    func testTokenDrivesAppAndModuleApproval() throws {
        try #"{"parlorServer":{"enabled":true,"port":11877}}"#.write(to: home.appendingPathComponent("settings.json"), atomically: true, encoding: .utf8)
        let dir = home.appendingPathComponent("modules/quiet")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"name":"quiet","description":"Does nothing.","command":"python3","args":["server.py"]}"#.write(to: dir.appendingPathComponent("module.json"), atomically: true, encoding: .utf8)
        try "import sys, json\nfor l in sys.stdin:\n    m = json.loads(l)\n    if m.get('id') is None: continue\n    r = {'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'quiet','version':'1'}} if m['method']=='initialize' else {'tools':[]}\n    print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}), flush=True)\n".write(to: dir.appendingPathComponent("server.py"), atomically: true, encoding: .utf8)
        app.launchEnvironment["PARLOR_ROUTE"] = "system"
        app.launch()

        // Module: listed, waiting for approval.
        let approve = app.buttons["approve-quiet"]
        XCTAssertTrue(approve.waitForExistence(timeout: 15), "unapproved module offers Approve")
        approve.click()
        XCTAssertTrue(app.buttons["revoke-quiet"].waitForExistence(timeout: 10), "approved")

        // Token: create in the sheet, read the secret back from the screen.
        app.buttons["new-token"].click()
        let name = app.textFields["token-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.click(); name.typeText("ui-test")
        app.buttons["create-token"].click()
        let secretText = app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'plt_'")).firstMatch
        XCTAssertTrue(secretText.waitForExistence(timeout: 5), "secret shown once")
        let secret = secretText.value as? String ?? ""
        XCTAssertTrue(secret.hasPrefix("plt_"))
        app.buttons["Done"].click()

        func rpc(_ method: String, _ params: [String: Any], token: String) throws -> (Int, [String: Any]) {
            var req = URLRequest(url: URL(string: "http://127.0.0.1:11877/mcp")!)
            req.httpMethod = "POST"
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": 1, "method": method, "params": params])
            var result: (Int, [String: Any]) = (0, [:])
            let done = expectation(description: "rpc")
            URLSession.shared.dataTask(with: req) { d, r, _ in
                result = ((r as? HTTPURLResponse)?.statusCode ?? 0, (d.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:])
                done.fulfill()
            }.resume()
            wait(for: [done], timeout: 10)
            return result
        }
        XCTAssertEqual(try rpc("tools/list", [:], token: "plt_nope").0, 401)
        let (s1, r1) = try rpc("tools/call", ["name": "conversations.create", "arguments": ["title": "From outside", "agents": ["mabel", "pip"], "scheduler": "roundRobin", "max_turns": 2]], token: secret)
        XCTAssertEqual(s1, 200)
        let text = (((r1["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String) ?? ""
        let id = (try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])?["id"] as? String ?? ""
        XCTAssertFalse(id.isEmpty, text)
        // The room appears in the app's sidebar, and a message posted over HTTP is answered by both agents.
        XCTAssertTrue(app.staticTexts["From outside"].firstMatch.waitForExistence(timeout: 10), "room shows up live")
        _ = try rpc("tools/call", ["name": "conversations.post", "arguments": ["id": id, "text": "hello from outside", "wait": true]], token: secret)
        app.staticTexts["From outside"].firstMatch.click()
        if !app.staticTexts["hello from outside"].firstMatch.waitForExistence(timeout: 10) {
            print("TEXTS:", app.staticTexts.allElementsBoundByIndex.map { ($0.value as? String) ?? $0.label }.joined(separator: " | "))
        }
        XCTAssertTrue(app.staticTexts["hello from outside"].exists)
        let replies = app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'"))
        XCTAssertGreaterThanOrEqual(replies.count, 2)
    }
}
