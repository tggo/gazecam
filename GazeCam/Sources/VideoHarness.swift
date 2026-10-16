import AVFoundation
import CoreImage

/// `GazeCam --video in.mov out.mp4 [--follow 0..1]` — runs the gaze effect over a whole clip (state carries
/// across frames, like live) and writes the side-by-side debug capture. For offline A/B on a recorded session.
enum VideoHarness {
    static func runIfRequested() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--video"), args.count > i + 2 else { return }
        let asset = AVURLAsset(url: URL(fileURLWithPath: args[i + 1]))
        let outURL = URL(fileURLWithPath: args[i + 2])
        guard let reader = try? AVAssetReader(asset: asset), let track = asset.tracks(withMediaType: .video).first else { print("cannot read"); exit(1) }
        let ro = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(ro); reader.startReading()

        let gaze = GazeEffect()
        if let j = args.firstIndex(of: "--follow"), args.count > j + 1, let f = Float(args[j + 1]) { gaze.followGaze = f }
        let cap = DebugCapture()
        guard let tmp = cap.startRecording() else { print("writer failed"); exit(1) }
        let ctx = CIContext()
        let w = CameraPipeline.width, h = CameraPipeline.height
        var n = 0
        while let sb = ro.copyNextSampleBuffer(), let src = CMSampleBufferGetImageBuffer(sb) {
            var img = CIImage(cvPixelBuffer: src)
            let s = max(CGFloat(w) / img.extent.width, CGFloat(h) / img.extent.height)
            img = img.transformed(by: CGAffineTransform(scaleX: s, y: s))
            img = img.transformed(by: CGAffineTransform(translationX: (CGFloat(w) - img.extent.width) / 2 - img.extent.minX,
                                                        y: (CGFloat(h) - img.extent.height) / 2 - img.extent.minY))
            var raw: CVPixelBuffer?, dst: CVPixelBuffer?
            let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &raw)
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &dst)
            ctx.render(img, to: raw!); ctx.render(img, to: dst!)
            gaze.apply(to: dst!)
            var st = CameraPipeline.Stats(); st.fps = 30
            // The writer drops frames when not ready; offline we can wait.
            cap.append(raw: raw!, processed: dst!, time: CMSampleBufferGetPresentationTimeStamp(sb), debug: args.contains("--clean") ? GazeEffect.Debug() : gaze.debug, stats: st)
            Thread.sleep(forTimeInterval: 0.005)
            n += 1
        }
        let sem = DispatchSemaphore(value: 0)
        cap.stopRecording(discard: false) { url in
            if let url { try? FileManager.default.removeItem(at: outURL); try? FileManager.default.moveItem(at: url, to: outURL) }
            sem.signal()
        }
        sem.wait()
        _ = tmp
        print("frames \(n) → \(outURL.path)"); exit(0)
    }
}
