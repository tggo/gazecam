import AVFoundation
import SwiftUI
import SystemExtensions

@main
struct GazeCamApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        MenuBarExtra("GazeCam", systemImage: model.running ? "eye.fill" : "eye") {
            MenuView().environmentObject(model)
        }
        .menuBarExtraStyle(.window)
    }
}

@MainActor
final class AppModel: NSObject, ObservableObject, OSSystemExtensionRequestDelegate {
    @Published var status = "Готово"
    @Published var extStatus = ""
    @Published var running = false
    @Published var cameras: [AVCaptureDevice] = CameraPipeline.cameras()
    @Published var cameraID: String = ""
    @Published var effectState: [String: Bool] = [:]

    let pipeline = CameraPipeline()
    private let extID = "dev.iamhero.GazeCam.Extension"

    override init() {
        super.init()
        cameraID = cameras.first?.uniqueID ?? ""
        for e in pipeline.effects { effectState[e.id] = e.enabled }
        pipeline.onStatus = { [weak self] s in Task { @MainActor in self?.status = s } }
        // Activation is idempotent; doing it on launch also picks up a rebuilt extension.
        installExtension()
    }

    func setEffect(_ id: String, _ on: Bool) {
        pipeline.effects.first { $0.id == id }?.enabled = on
        effectState[id] = on
    }

    func toggleRun() {
        if running { pipeline.stop(); running = false; return }
        guard let cam = cameras.first(where: { $0.uniqueID == cameraID }) ?? cameras.first else { status = "Немає камери"; return }
        AVCaptureDevice.requestAccess(for: .video) { ok in
            Task { @MainActor in
                guard ok else { self.status = "Нема доступу до камери"; return }
                self.pipeline.start(camera: cam)
                self.running = true
            }
        }
    }

    func installExtension() {
        let r = OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: extID, queue: .main)
        r.delegate = self
        OSSystemExtensionManager.shared.submitRequest(r)
        extStatus = "Встановлюю…"
    }

    func uninstallExtension() {
        let r = OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: extID, queue: .main)
        r.delegate = self
        OSSystemExtensionManager.shared.submitRequest(r)
        extStatus = "Видаляю…"
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, actionForReplacingExtension existing: OSSystemExtensionProperties,
                             withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction { .replace }

    nonisolated func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        Task { @MainActor in self.extStatus = "Дозволь у System Settings → General → Login Items & Extensions → Camera Extensions" }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
        Task { @MainActor in self.extStatus = result == .completed ? "Розширення OK" : "Потрібен перезапуск" }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        Task { @MainActor in self.extStatus = "Помилка: \(error.localizedDescription)" }
    }
}

struct MenuView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("GazeCam").font(.headline)
            Picker("Камера", selection: $model.cameraID) {
                ForEach(model.cameras, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
            }
            .disabled(model.running)

            Button(model.running ? "Зупинити" : "Старт") { model.toggleRun() }
                .keyboardShortcut(.defaultAction)
            Text(model.status).font(.caption).foregroundStyle(.secondary)

            Divider()
            Text("Ефекти").font(.subheadline)
            ForEach(model.pipeline.effects, id: \.id) { e in
                Toggle(e.title, isOn: Binding(get: { model.effectState[e.id] ?? false }, set: { model.setEffect(e.id, $0) }))
            }

            Divider()
            HStack {
                Button("Встановити камеру") { model.installExtension() }
                Button("Видалити") { model.uninstallExtension() }
            }
            if !model.extStatus.isEmpty { Text(model.extStatus).font(.caption).foregroundStyle(.secondary) }

            Divider()
            Button("Вийти") { NSApp.terminate(nil) }
        }
        .padding(14)
        .frame(width: 300)
    }
}
