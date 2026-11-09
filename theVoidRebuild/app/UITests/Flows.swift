import XCTest

/// Simulator flows. Real UI and real SwiftData storage; enrichment is seeded because MLX and SpeechAnalyzer
/// can't run in the simulator. Capture/recording flows live in the Mac harness and on the phone.
final class Flows: XCTestCase {
    override func setUp() { continueAfterFailure = false }

    private func launch(_ args: [String]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = args
        app.launch()
        return app
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = ProcessInfo.processInfo.environment["VOID_SHOTS"] {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: "\(dir)/\(name).png"))
        }
        let a = XCTAttachment(data: png, uniformTypeIdentifier: "public.png"); a.name = name; a.lifetime = .keepAlways
        add(a)
    }

    // F4 edit, F8 delete (and the privacy toggle on the card), through the real Void tab.
    func testVoidListEditShareDelete() {
        let app = launch(["-seed", "-tab", "1"])
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        XCTAssertEqual(app.cells.count, 4)
        shot("void-list")

        app.cells.firstMatch.tap()                                          // newest: "Big launch day", shared
        let title = app.textViews["titleField"].exists ? app.textViews["titleField"] : app.textFields["titleField"]
        XCTAssertTrue(title.waitForExistence(timeout: 5))
        XCTAssertEqual(title.value as? String, "Big launch day")
        title.tap()
        title.press(forDuration: 1.0)
        app.menuItems["Select All"].tap()
        title.typeText("Renamed by me\n")
        shot("card")

        app.buttons["tag_low_unpleasant"].tap()                             // add a second tag
        let toggle = app.switches.firstMatch
        XCTAssertEqual(toggle.value as? String, "1", "seeded entry is shared")
        toggle.tap()
        XCTAssertEqual(toggle.value as? String, "0", "flipped to private")

        app.navigationBars.buttons.firstMatch.tap()                         // back to the list
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'Renamed by me'")).firstMatch.waitForExistence(timeout: 5),
                      "title edit did not persist to the list")

        app.cells.firstMatch.tap()
        XCTAssertEqual(app.switches.firstMatch.value as? String, "0", "visibility did not persist")
        app.buttons["Delete message"].tap()
        app.sheets.buttons["Delete"].firstMatch.tap()
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 5))
        XCTAssertEqual(app.cells.count, 3, "delete did not remove the entry")
        shot("void-after-delete")
    }

    // D5: private on first use, then remembers the last choice across launches.
    func testVisibilityDefaultsPrivateThenRemembers() {
        var app = launch([])
        let toggle = app.buttons["visibilityToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10))
        XCTAssertEqual(toggle.label, "private", "first use must be private")
        toggle.tap()
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "shared with friends")
        shot("capture-shared")
        app.terminate()
        app = launch([])
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "shared with friends", "last choice not remembered")
        app.buttons["visibilityToggle"].tap()   // leave the simulator in the default state
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "private")
    }

    // F7 (display side): friends' dots appear with tags and time only, never content.
    func testSocialShowsDotsNotContent() {
        let app = launch(["-seed", "-tab", "2"])
        XCTAssertTrue(app.otherElements["friendRow"].firstMatch.waitForExistence(timeout: 10)
                      || app.staticTexts["Maya"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Maya"].exists && app.staticTexts["Dev"].exists)
        shot("social")
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'shipped it'")).firstMatch.exists,
                       "transcript text must never appear in Social")
    }

    // MARK: Photo mode and settings. The simulator has no camera, so `-fixtureCamera` supplies the two images;
    // everything after the shutter (storage, list, card, settings) is the real app. No mic here either, so a "hold"
    // exercises the snap but not the voice joining it (that is in the Mac harness).

    private func el(_ app: XCUIApplication, _ id: String) -> XCUIElement { app.descendants(matching: .any)[id].firstMatch }

    /// A Form switch's frame spans the row; its control is at the trailing edge.
    private func flip(_ toggle: XCUIElement) { toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.92, dy: 0.5)).tap() }

    private func openCamera(_ app: XCUIApplication) {
        let glyph = app.buttons["cameraGlyph"]
        XCTAssertTrue(glyph.waitForExistence(timeout: 10))
        app.swipeLeft()                                                    // the gesture; falls back to the glyph
        if !el(app, "shutterCircle").waitForExistence(timeout: 3) { glyph.tap() }
        XCTAssertTrue(el(app, "shutterCircle").waitForExistence(timeout: 5), "camera page did not open")
    }

    // F10: swipe to the camera, tap the circle -> one new photo-only message with both images, in the Void.
    func testSnapMakesAPhotoMessage() {
        let app = launch(["-seed", "-fixtureCamera", "-tab", "0"])
        openCamera(app)
        shot("camera")
        el(app, "shutterCircle").tap()
        XCTAssertTrue(el(app, "snappedStrip").waitForExistence(timeout: 10), "no snapped images after the tap")
        XCTAssertTrue(el(app, "thumbBack").exists)
        XCTAssertTrue(el(app, "thumbFront").exists, "front image missing")
        shot("camera-snapped")

        app.tabBars.buttons["Void"].tap()
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        XCTAssertEqual(app.cells.count, 5, "one new message expected (4 seeded + 1 photo)")
        app.cells.firstMatch.tap()
        XCTAssertTrue(el(app, "entryPhotos").waitForExistence(timeout: 5), "card does not show the photos")
        XCTAssertFalse(app.textViews["transcriptEditor"].exists, "a photo-only message has no transcript")
        shot("card-photo")
    }

    // F10/F8: the ✕ after a snap discards the message, and nothing is left in the Void.
    func testDiscardRemovesThePhotoMessage() {
        let app = launch(["-seed", "-fixtureCamera", "-tab", "0"])
        openCamera(app)
        el(app, "shutterCircle").tap()
        XCTAssertTrue(app.buttons["discardPhoto"].waitForExistence(timeout: 10))
        app.buttons["discardPhoto"].tap()
        XCTAssertFalse(el(app, "snappedStrip").waitForExistence(timeout: 2))
        app.tabBars.buttons["Void"].tap()
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        XCTAssertEqual(app.cells.count, 4, "discard left a message behind")
    }

    // F11 (snap half): touch and keep holding still snaps once, on touch, into exactly one message.
    func testHoldSnapsOneMessage() {
        let app = launch(["-seed", "-fixtureCamera", "-tab", "0"])
        openCamera(app)
        el(app, "shutterCircle").press(forDuration: 1.5)
        XCTAssertTrue(el(app, "snappedStrip").waitForExistence(timeout: 10))
        app.tabBars.buttons["Void"].tap()
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        XCTAssertEqual(app.cells.count, 5, "a hold must still be exactly one message")
    }

    // F12: "add photo" from a card puts the photo on that message and adds no message.
    func testAddPhotoToAnExistingMessage() {
        let app = launch(["-seed", "-fixtureCamera", "-tab", "1"])
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        app.cells.element(boundBy: 1).tap()                                 // "Quiet evening", voice only
        XCTAssertFalse(el(app, "entryPhotos").exists)
        app.buttons["addPhotoButton"].tap()
        XCTAssertTrue(el(app, "shutterCircle").waitForExistence(timeout: 5), "camera did not open over the card")
        shot("add-photo")
        el(app, "shutterCircle").tap()
        XCTAssertTrue(el(app, "entryPhotos").waitForExistence(timeout: 10), "photo did not join the message")
        XCTAssertTrue(app.textViews["transcriptEditor"].exists, "the voice part must survive")
        app.navigationBars.buttons.firstMatch.tap()
        XCTAssertEqual(app.cells.count, 4, "add photo must not create a message")
    }

    // F13: settings persist across launches and do what they say (back-only means no front image).
    func testSettingsPersistAndApply() {
        var app = launch(["-seed", "-fixtureCamera", "-tab", "1"])
        app.buttons["settingsGear"].tap()
        let backOnly = app.switches["backOnlyToggle"]
        XCTAssertTrue(backOnly.waitForExistence(timeout: 5))
        XCTAssertEqual(backOnly.value as? String, "0", "default is front and back")
        shot("settings")
        flip(backOnly)
        XCTAssertEqual(backOnly.value as? String, "1")
        app.buttons["Done"].tap()
        app.terminate()

        app = launch(["-seed", "-fixtureCamera", "-tab", "1"])
        app.buttons["settingsGear"].tap()
        XCTAssertEqual(app.switches["backOnlyToggle"].value as? String, "1", "setting did not persist")
        app.buttons["Done"].tap()
        app.tabBars.buttons["Capture"].tap()
        openCamera(app)
        el(app, "shutterCircle").tap()
        XCTAssertTrue(el(app, "snappedStrip").waitForExistence(timeout: 10))
        XCTAssertFalse(el(app, "thumbFront").exists, "back-only must not keep a front image")

        // Leave the simulator in the default state for the next run.
        app.tabBars.buttons["Void"].tap()
        app.buttons["settingsGear"].tap()
        flip(app.switches["backOnlyToggle"])
        XCTAssertEqual(app.switches["backOnlyToggle"].value as? String, "0")
    }
}
