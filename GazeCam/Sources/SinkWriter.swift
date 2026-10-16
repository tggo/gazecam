import CoreMediaIO
import CoreMedia
import Foundation

/// Finds the GazeCam virtual device through the CoreMediaIO C API and enqueues frames into its sink stream.
final class SinkWriter {
    private var deviceID: CMIOObjectID = 0
    private var sinkStreamID: CMIOStreamID = 0
    private var queue: CMSimpleQueue?
    private(set) var connected = false

    func connect(deviceName: String = "GazeCam") -> Bool {
        if connected { return true }
        // Allow the process to see devices from camera extensions.
        var allow: UInt32 = 1
        var allowAddr = CMIOObjectPropertyAddress(mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyAllowScreenCaptureDevices),
                                                  mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                                                  mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
        CMIOObjectSetPropertyData(CMIOObjectID(kCMIOObjectSystemObject), &allowAddr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &allow)

        guard let dev = devices().first(where: { name(of: $0) == deviceName }) else { return false }
        let streams = streamIDs(of: dev)
        // Streams come in the order the extension added them: [source, sink].
        guard streams.count >= 2 else { return false }
        let sink = streams[1]

        var q: Unmanaged<CMSimpleQueue>?
        let err = CMIOStreamCopyBufferQueue(sink, { _, _, _ in }, nil, &q)
        guard err == noErr, let q else { NSLog("GazeCam: CMIOStreamCopyBufferQueue %d", err); return false }
        queue = q.takeRetainedValue()
        guard CMIODeviceStartStream(dev, sink) == noErr else { NSLog("GazeCam: CMIODeviceStartStream failed"); return false }
        deviceID = dev
        sinkStreamID = sink
        connected = true
        return true
    }

    func disconnect() {
        guard connected else { return }
        CMIODeviceStopStream(deviceID, sinkStreamID)
        queue = nil
        connected = false
    }

    func enqueue(_ sb: CMSampleBuffer) {
        guard let queue else { return }
        // Drop instead of blocking when the extension is behind.
        guard CMSimpleQueueGetCount(queue) < CMSimpleQueueGetCapacity(queue) else { return }
        let ptr = Unmanaged.passRetained(sb).toOpaque()
        if CMSimpleQueueEnqueue(queue, element: ptr) != noErr {
            Unmanaged<CMSampleBuffer>.fromOpaque(ptr).release()
        }
    }

    // MARK: CMIO helpers

    private func addr(_ sel: Int) -> CMIOObjectPropertyAddress {
        CMIOObjectPropertyAddress(mSelector: CMIOObjectPropertySelector(sel),
                                  mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                                  mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
    }

    private func ids(of obj: CMIOObjectID, selector: Int) -> [CMIOObjectID] {
        var a = addr(selector)
        var size: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(obj, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
        var out = [CMIOObjectID](repeating: 0, count: Int(size) / MemoryLayout<CMIOObjectID>.size)
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(obj, &a, 0, nil, size, &used, &out) == noErr else { return [] }
        return out
    }

    private func devices() -> [CMIOObjectID] { ids(of: CMIOObjectID(kCMIOObjectSystemObject), selector: kCMIOHardwarePropertyDevices) }
    private func streamIDs(of dev: CMIOObjectID) -> [CMIOStreamID] { ids(of: dev, selector: kCMIODevicePropertyStreams) }

    private func name(of obj: CMIOObjectID) -> String? {
        var a = addr(kCMIOObjectPropertyName)
        var cf: Unmanaged<CFString>?
        var used: UInt32 = 0
        let size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard CMIOObjectGetPropertyData(obj, &a, 0, nil, size, &used, &cf) == noErr, let cf else { return nil }
        return cf.takeRetainedValue() as String
    }
}
