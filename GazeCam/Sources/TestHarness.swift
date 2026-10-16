import AppKit
import CoreImage

/// `GazeCam --test in.png out.png [effect-id]` — runs one effect on a still image and exits.
/// Used to check an effect without a camera or the virtual device.
enum TestHarness {
    static func runIfRequested() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--test"), args.count > i + 2 else { return }
        let input = URL(fileURLWithPath: args[i + 1]), output = URL(fileURLWithPath: args[i + 2])
        let effectID = args.count > i + 3 ? args[i + 3] : "gaze"
        guard let ci = CIImage(contentsOf: input) else { print("cannot read \(input.path)"); exit(1) }

        let w = CameraPipeline.width, h = CameraPipeline.height
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
        let ctx = CIContext()
        let s = max(CGFloat(w) / ci.extent.width, CGFloat(h) / ci.extent.height)
        var img = ci.transformed(by: CGAffineTransform(scaleX: s, y: s))
        img = img.transformed(by: CGAffineTransform(translationX: (CGFloat(w) - img.extent.width) / 2 - img.extent.minX,
                                                    y: (CGFloat(h) - img.extent.height) / 2 - img.extent.minY))
        ctx.render(img, to: pb!)

        if effectID != "none" {
            let effect: Effect = effectID == "horns" ? HornsEffect() : GazeEffect()
            effect.enabled = true
            var warm: CVPixelBuffer? // warm-up on a copy: Vision + Core ML first-run costs
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &warm)
            ctx.render(img, to: warm!)
            effect.apply(to: warm!)
            let t0 = Date()
            let n = effectID == "gaze" ? 1 : 3 // gaze warps the frame in place, so one timed pass
            for _ in 0..<n { effect.apply(to: pb!) }
            print(String(format: "%@ applied, %.1f ms/frame", effectID, Date().timeIntervalSince(t0) * 1000 / Double(n)))
        }

        let out = CIImage(cvPixelBuffer: pb!)
        try? ctx.writePNGRepresentation(of: out, to: output, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        exit(0)
    }
}
