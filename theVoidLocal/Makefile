SHELL := /bin/bash

IOS_PROJECT := theVoid/theVoid.xcodeproj
IOS_SCHEME := theVoid
IOS_DESTINATION ?= platform=iOS Simulator,name=iPhone 16 Pro

.PHONY: help ios-build ios-test

help:
	@echo "theVoid local targets"
	@echo "  make ios-build    # build iOS app for simulator"
	@echo "  make ios-test     # run iOS tests on simulator"

ios-build:
	@xcodebuild -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" -destination "$(IOS_DESTINATION)" build

ios-test:
	@xcodebuild -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" -destination "$(IOS_DESTINATION)" test
