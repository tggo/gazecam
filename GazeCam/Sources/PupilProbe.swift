import AVFoundation
import Vision

/// `GazeCam --pupils capture.mp4 out.csv` — per-frame pupil offset inside each eye, from the ORIGINAL (left) half
/// of a debug capture. Measures whether Vision's pupil points carry the gaze signal (reading sweeps) at all.
enum PupilProbe {
    static func runIfRequested() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--pupils"), args.count > i + 2 else { return }
        let asset = AVURLAsset(url: URL(fileURLWithPath: args[i + 1]))
        guard let reader = try? AVAssetReader(asset: asset),
              let track = asset.tracks(withMediaType: .video).first else { print("cannot read"); exit(1) }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out); reader.startReading()
        var csv = "t,lx,ly,rx,ry\n"
        let req = VNDetectFaceLandmarksRequest()
        while let sb = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) {
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            let half = CGRect(x: 0, y: 0, width: 0.5, height: 1) // original half
            if args.contains("--right") { req.regionOfInterest = CGRect(x: 0.5, y: 0, width: 0.5, height: 1) } else if !args.contains("--full") { req.regionOfInterest = half }
            try? VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([req])
            guard let lm = req.results?.first?.landmarks, let le = lm.leftEye, let re = lm.rightEye,
                  let lp = lm.leftPupil, let rp = lm.rightPupil else { csv += "\(t),,,,\n"; continue }
            func off(_ eye: VNFaceLandmarkRegion2D, _ pupil: VNFaceLandmarkRegion2D) -> (Double, Double) {
                let e = eye.normalizedPoints, p = pupil.normalizedPoints[0]
                let l = e.min { $0.x < $1.x }!, r = e.max { $0.x < $1.x }!
                let w = max(r.x - l.x, 0.001)
                let c = CGPoint(x: (l.x + r.x) / 2, y: (l.y + r.y) / 2)
                return (Double((p.x - c.x) / w), Double((p.y - c.y) / w))
            }
            let a = off(le, lp), b = off(re, rp)
            csv += String(format: "%.3f,%.4f,%.4f,%.4f,%.4f\n", t, a.0, a.1, b.0, b.1)
        }
        try? csv.write(toFile: args[i + 2], atomically: true, encoding: .utf8)
        print("wrote", args[i + 2]); exit(0)
    }
}
