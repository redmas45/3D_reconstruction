import bpy


CYCLES_RENDER_ENGINE = "CYCLES"
SUPPORTED_CYCLES_COMPUTE_DEVICES = frozenset({"CUDA", "OPTIX"})
DEFAULT_CYCLES_SAMPLES = 16


def configure_cycles_render(scene: bpy.types.Scene, render_contract: dict) -> list[str]:
    compute_device = str(render_contract.get("cycles_compute_device", "CUDA")).upper()
    if compute_device not in SUPPORTED_CYCLES_COMPUTE_DEVICES:
        raise ValueError(f"Unsupported Cycles compute device: {compute_device}")
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = compute_device
    preferences.refresh_devices()
    enabled_device_names = _enable_compute_devices(preferences.devices, compute_device)
    if not enabled_device_names:
        raise RuntimeError(f"Cycles could not find an available {compute_device} device")
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(render_contract.get("cycles_samples", DEFAULT_CYCLES_SAMPLES))
    scene.cycles.use_denoising = bool(render_contract.get("cycles_use_denoising", True))
    return enabled_device_names


def _enable_compute_devices(devices: object, compute_device: str) -> list[str]:
    enabled_device_names: list[str] = []
    for device in devices:
        use_device = device.type == compute_device
        device.use = use_device
        if use_device:
            enabled_device_names.append(str(device.name))
    return enabled_device_names
