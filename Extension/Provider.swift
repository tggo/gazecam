import CoreMediaIO
import CoreVideo
import Foundation
import IOKit.audio
import os.log

// Shared with the app: the app finds the device by this UID and feeds frames into its sink stream.
let kDeviceUID = "GazeCam-Device"
let kWidth: Int32 = 1280
let kHeight: Int32 = 720
let kFPS = 30

private let log = Logger(subsystem: "dev.iamhero.GazeCam.Extension", category: "provider")

// MARK: - Device

final class DeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var sourceStream: SourceStream!
    private var sinkStream: SinkStream!

    let format: CMFormatDescription
    private let pool: CVPixelBufferPool
    private let poolAux: NSDictionary = [kCVPixelBufferPoolAllocationThresholdKey: 4]

    private var sourceCount = 0
    private var sinkClient: CMIOExtensionClient?
    private var sinkRunning = false
    private var lastSinkFrame: UInt64 = 0

    private let queue = DispatchQueue(label: "gazecam.device")
    private var idleTimer: DispatchSourceTimer?

    init(localizedName: String) {
        var fmt: CMFormatDescription?
        CMVideoFormatDescriptionCreate(allocator: kCFAllocatorDefault, codecType: kCVPixelFormatType_32BGRA,
                                       width: kWidth, height: kHeight, extensions: nil, formatDescriptionOut: &fmt)
        format = fmt!
        let attrs: NSDictionary = [
            kCVPixelBufferWidthKey: kWidth, kCVPixelBufferHeightKey: kHeight,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as NSDictionary,
        ]
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs, &p)
        pool = p!
        super.init()

        device = CMIOExtensionDevice(localizedName: localizedName, deviceID: UUID(uuidString: "6A3C1C9E-2B7D-4F0E-9C1A-0A2E0CA00001")!,
                                     legacyDeviceID: kDeviceUID, source: self)

        let dur = CMTime(value: 1, timescale: Int32(kFPS))
        let streamFormat = CMIOExtensionStreamFormat(formatDescription: format, maxFrameDuration: dur, minFrameDuration: dur, validFrameDurations: nil)
        sourceStream = SourceStream(localizedName: "GazeCam.Video", streamID: UUID(), format: streamFormat, device: self)
        sinkStream = SinkStream(localizedName: "GazeCam.Sink", streamID: UUID(), format: streamFormat, device: self)
        do {
            try device.addStream(sourceStream.stream)
            try device.addStream(sinkStream.stream)
        } catch {
            fatalError("addStream: \(error)")
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> { [.deviceTransportType, .deviceModel] }

    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionDeviceProperties {
        let p = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) { p.transportType = kIOAudioDeviceTransportTypeVirtual }
        if properties.contains(.deviceModel) { p.model = "GazeCam" }
        return p
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}

    // Source (what Meet reads). When the app is not feeding frames, emit a placeholder.
    func startSource() {
        queue.async { [self] in
            sourceCount += 1
            guard idleTimer == nil else { return }
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now(), repeating: 1.0 / Double(kFPS))
            t.setEventHandler { [weak self] in self?.emitPlaceholderIfIdle() }
            t.resume()
            idleTimer = t
        }
    }

    func stopSource() {
        queue.async { [self] in
            sourceCount = max(0, sourceCount - 1)
            if sourceCount == 0 { idleTimer?.cancel(); idleTimer = nil }
        }
    }

    private func emitPlaceholderIfIdle() {
        let now = mach_absolute_time()
        if lastSinkFrame != 0, nanos(now - lastSinkFrame) < 500_000_000 { return }
        var pb: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, poolAux, &pb) == kCVReturnSuccess, let pb else { return }
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        // Dark slate with a moving bar so it's obvious the camera is alive but the app is not feeding it.
        memset(base, 0x30, rowBytes * Int(kHeight))
        let barX = Int(nanos(now) / 10_000_000) % Int(kWidth)
        for y in 0..<Int(kHeight) {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt32.self)
            for x in barX..<min(barX + 20, Int(kWidth)) { row[x] = 0xFF_FF_80_20 }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        send(pixelBuffer: pb, hostTime: now)
    }

    private func send(pixelBuffer pb: CVPixelBuffer, hostTime: UInt64) {
        var sb: CMSampleBuffer?
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: Int32(kFPS)),
                                        presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
                                        decodeTimeStamp: .invalid)
        guard CMSampleBufferCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pb, dataReady: true,
                                                 makeDataReadyCallback: nil, refcon: nil, formatDescription: format,
                                                 sampleTiming: &timing, sampleBufferOut: &sb) == noErr, let sb else { return }
        sourceStream.stream.send(sb, discontinuity: [], hostTimeInNanoseconds: nanos(hostTime))
    }

    // Sink (what the app writes).
    func authorizeSink(_ client: CMIOExtensionClient) { queue.sync { sinkClient = client } }

    func startSink() {
        queue.async { [self] in
            sinkRunning = true
            consumeNext()
        }
    }

    func stopSink() { queue.async { [self] in sinkRunning = false } }

    private func consumeNext() {
        guard sinkRunning, let client = sinkClient else { return }
        sinkStream.stream.consumeSampleBuffer(from: client) { [weak self] sb, seq, _, _, err in
            guard let self else { return }
            self.queue.async {
                if let sb {
                    let now = mach_absolute_time()
                    self.lastSinkFrame = now
                    if self.sourceCount > 0, let pb = CMSampleBufferGetImageBuffer(sb) {
                        self.send(pixelBuffer: pb, hostTime: now)
                    }
                    let out = CMIOExtensionScheduledOutput(sequenceNumber: seq, hostTimeInNanoseconds: self.nanos(now))
                    self.sinkStream.stream.notifyScheduledOutputChanged(out)
                    self.consumeNext()
                } else {
                    if let err { log.debug("consume: \(err.localizedDescription)") }
                    self.queue.asyncAfter(deadline: .now() + 1.0 / Double(kFPS * 2)) { self.consumeNext() }
                }
            }
        }
    }

    private static let timebase: mach_timebase_info_data_t = { var t = mach_timebase_info_data_t(); mach_timebase_info(&t); return t }()
    private func nanos(_ ticks: UInt64) -> UInt64 { ticks * UInt64(Self.timebase.numer) / UInt64(Self.timebase.denom) }
}

// MARK: - Streams

final class SourceStream: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    private let format: CMIOExtensionStreamFormat
    private unowned let device: DeviceSource

    init(localizedName: String, streamID: UUID, format: CMIOExtensionStreamFormat, device: DeviceSource) {
        self.format = format
        self.device = device
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName, streamID: streamID, direction: .source, clockType: .hostTime, source: self)
    }

    var formats: [CMIOExtensionStreamFormat] { [format] }
    var availableProperties: Set<CMIOExtensionProperty> { [.streamActiveFormatIndex, .streamFrameDuration] }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let p = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { p.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) { p.frameDuration = CMTime(value: 1, timescale: Int32(kFPS)) }
        return p
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}
    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }
    func startStream() throws { device.startSource() }
    func stopStream() throws { device.stopSource() }
}

final class SinkStream: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    private let format: CMIOExtensionStreamFormat
    private unowned let device: DeviceSource

    init(localizedName: String, streamID: UUID, format: CMIOExtensionStreamFormat, device: DeviceSource) {
        self.format = format
        self.device = device
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName, streamID: streamID, direction: .sink, clockType: .hostTime, source: self)
    }

    var formats: [CMIOExtensionStreamFormat] { [format] }
    var availableProperties: Set<CMIOExtensionProperty> {
        [.streamActiveFormatIndex, .streamFrameDuration, .streamSinkBufferQueueSize,
         .streamSinkBuffersRequiredForStartup, .streamSinkBufferUnderrunCount, .streamSinkEndOfData]
    }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let p = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { p.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) { p.frameDuration = CMTime(value: 1, timescale: Int32(kFPS)) }
        if properties.contains(.streamSinkBufferQueueSize) { p.sinkBufferQueueSize = 1 }
        if properties.contains(.streamSinkBuffersRequiredForStartup) { p.sinkBuffersRequiredForStartup = 1 }
        return p
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        device.authorizeSink(client)
        return true
    }

    func startStream() throws { device.startSink() }
    func stopStream() throws { device.stopSink() }
}

// MARK: - Provider

final class ProviderSource: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: DeviceSource!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = DeviceSource(localizedName: "GazeCam")
        do { try provider.addDevice(deviceSource.device) } catch { fatalError("addDevice: \(error)") }
    }

    func connect(to client: CMIOExtensionClient) throws {}
    func disconnect(from client: CMIOExtensionClient) {}
    var availableProperties: Set<CMIOExtensionProperty> { [.providerManufacturer] }

    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let p = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) { p.manufacturer = "iamhero" }
        return p
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
