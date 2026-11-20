// Draws the app icon: the capture circle on black. Opaque 1024x1024 (App Store icons can't have alpha).
// Run: swift tools/make-icon.swift Assets.xcassets/AppIcon.appiconset/icon-1024.png
import AppKit
import CoreGraphics

let size = 1024
let space = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
let center = CGPoint(x: size / 2, y: size / 2)
// A faint halo, then the circle.
let colors = [CGColor(red: 1, green: 1, blue: 1, alpha: 0.18), CGColor(red: 1, green: 1, blue: 1, alpha: 0)] as CFArray
let gradient = CGGradient(colorsSpace: space, colors: colors, locations: [0, 1])!
ctx.drawRadialGradient(gradient, startCenter: center, startRadius: 150, endCenter: center, endRadius: 470, options: [])
ctx.setFillColor(CGColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1))
ctx.fillEllipse(in: CGRect(x: center.x - 190, y: center.y - 190, width: 380, height: 380))
let rep = NSBitmapImageRep(cgImage: ctx.makeImage()!)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
