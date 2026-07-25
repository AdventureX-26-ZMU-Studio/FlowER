#!/usr/bin/env bash
# Synchronize and verify the locked Galaxea A1Z runtime, then configure the
# platform CAN transport. This is the one-command A1Z host setup.
set -euo pipefail

DEPENDENCY_GROUP="galaxea-a1z"
GB10_TORCH_VERSION="2.10.0+cu130"
GB10_TORCHVISION_VERSION="0.25.0+cu130"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
CAN_SETUP_SCRIPT="$SCRIPT_DIR/setup_a1z_can.sh"

usage() {
    cat <<EOF
Usage: $0 [--sdk-only] [--with-lerobot]

Synchronize the locked Galaxea A1Z runtime into the DimOS virtual environment,
repair OpenCV if a conflicting wheel previously damaged it, then configure and
test the platform CAN adapter.

  --sdk-only      Install/verify Python dependencies without checking hardware.
  --with-lerobot  Also install dataset export, LeRobot training, and inference.

Run this command as your normal user. Linux requests sudo only for SocketCAN;
macOS installs Homebrew libusb when it is missing.
EOF
}

sdk_only=false
with_lerobot=false
while (($#)); do
    case "$1" in
        --sdk-only) sdk_only=true ;;
        --with-lerobot) with_lerobot=true ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ((EUID == 0)); then
    cat >&2 <<EOF
Do not run this wrapper with sudo.

It synchronizes the Python runtime as your normal user and requests sudo itself
only for the SocketCAN setup:

  $0
EOF
    exit 1
fi

uv_bin="${A1Z_UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" ]]; then
    cat >&2 <<EOF
The A1Z setup requires 'uv', but it was not found.

Install uv, then rerun:
  $0
EOF
    exit 1
fi

gb10_host=false
gpu_name=""
if [[ "$(uname -s)" == Linux ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ "${gpu_name^^}" == *GB10* ]]; then
        gb10_host=true
        echo "Detected $gpu_name; preserving the CUDA 13.0 PyTorch runtime."
    fi
fi

sync_project() {
    local -a sync_args=(
        sync
        --locked
        --inexact
        --group "$DEPENDENCY_GROUP"
    )
    if [[ "$gb10_host" == true ]]; then
        # The universal lock resolves the CPU PyTorch build. GB10 needs the
        # CUDA 13.0 aarch64 wheels installed and verified below.
        sync_args+=(
            --no-install-package torch
            --no-install-package torchvision
        )
    fi
    if [[ "$with_lerobot" == true ]]; then
        sync_args+=(--extra learning --extra lerobot)
    fi
    sync_args+=("$@")
    (cd "$REPOSITORY_ROOT" && "$uv_bin" "${sync_args[@]}")
}

echo "Synchronizing the locked DimOS A1Z runtime..."
sync_project

python_bin="${DIMOS_PYTHON:-$REPOSITORY_ROOT/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: uv did not create the expected Python at: $python_bin" >&2
    exit 1
fi

verify_gb10_torch() {
    "$python_bin" - "$GB10_TORCH_VERSION" "$GB10_TORCHVISION_VERSION" <<'PY'
import sys

expected_torch, expected_torchvision = sys.argv[1:]

import torch
import torchvision

if torch.__version__ != expected_torch:
    raise SystemExit(
        f"expected torch {expected_torch}, found {torch.__version__}"
    )
if torchvision.__version__ != expected_torchvision:
    raise SystemExit(
        f"expected torchvision {expected_torchvision}, "
        f"found {torchvision.__version__}"
    )
if torch.version.cuda != "13.0":
    raise SystemExit(f"expected CUDA 13.0 runtime, found {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")

device_name = torch.cuda.get_device_name(0)
if "GB10" not in device_name.upper():
    raise SystemExit(f"CUDA device 0 is not GB10: {device_name}")

# Exercise a real CUDA kernel, not only driver/device discovery.
x = torch.randn((256, 256), device="cuda")
y = x @ x
conv = torch.nn.Conv2d(3, 8, kernel_size=3).cuda()
features = conv(torch.randn((1, 3, 32, 32), device="cuda"))
torch.cuda.synchronize()
if not torch.isfinite(y).all().item() or not torch.isfinite(features).all().item():
    raise SystemExit("GB10 CUDA verification produced non-finite values")
if not torch.backends.cudnn.is_available():
    raise SystemExit("cuDNN is unavailable")

print(
    f"torch={torch.__version__} torchvision={torchvision.__version__} "
    f"cuda={torch.version.cuda} device={device_name} "
    f"capability={torch.cuda.get_device_capability(0)} "
    f"cudnn={torch.backends.cudnn.version()}"
)
PY
}

if [[ "$gb10_host" == true ]]; then
    if ! gb10_torch_status="$(verify_gb10_torch 2>/dev/null)"; then
        gb10_find_links="${A1Z_GB10_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu130/}"
        echo "Installing the GB10 CUDA 13.0 PyTorch wheels..."
        echo "Using the CUDA 13 libraries shipped on the GB10 host."
        (
            cd /tmp
            # The PyTorch wheel metadata lists standalone NVIDIA packages that
            # duplicate the CUDA 13 toolkit already installed on GB10 systems.
            # Installing only the two cached/mirrored wheels avoids downloading
            # several gigabytes and keeps setup usable on mainland networks.
            "$uv_bin" --no-config pip install \
                --no-deps \
                --python "$python_bin" \
                --find-links "$gb10_find_links" \
                "torch==$GB10_TORCH_VERSION" \
                "torchvision==$GB10_TORCHVISION_VERSION"
        )
        gb10_torch_status="$(verify_gb10_torch)"
    fi
    echo "GB10 CUDA check passed: $gb10_torch_status"
fi

verify_sdk() {
    "$python_bin" - <<'PY'
import inspect
import sys

try:
    import a1z
    from a1z.robots.get_robot import get_a1z_robot
except Exception as exc:
    print(f"cannot import a1z: {exc}", file=sys.stderr)
    raise SystemExit(1) from None

try:
    parameters = inspect.signature(get_a1z_robot).parameters
except (TypeError, ValueError) as exc:
    print(f"cannot inspect a1z get_a1z_robot(): {exc}", file=sys.stderr)
    raise SystemExit(1) from None

if "with_gripper" not in parameters:
    print(
        "installed a1z SDK does not expose get_a1z_robot(with_gripper=...); "
        "the vendor gripper branch is required",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(a1z.__file__)
PY
}

verify_opencv() {
    "$python_bin" - <<'PY'
import cv2

required = ("VideoCapture", "VideoWriter", "cvtColor")
missing = [name for name in required if not hasattr(cv2, name)]
if missing:
    raise SystemExit(f"cv2 is missing runtime symbols: {', '.join(missing)}")
print(cv2.__version__)
PY
}

opencv_needs_reinstall=false
for conflicting_package in opencv-python opencv-python-headless; do
    if "$python_bin" - "$conflicting_package" <<'PY' >/dev/null 2>&1
from importlib import metadata
import sys

metadata.version(sys.argv[1])
PY
    then
        echo "Removing conflicting $conflicting_package wheel..."
        "$uv_bin" pip uninstall --python "$python_bin" "$conflicting_package"
        opencv_needs_reinstall=true
    fi
done

if [[ "$opencv_needs_reinstall" == true ]] || ! opencv_version="$(verify_opencv 2>/dev/null)"; then
    echo "Repairing the OpenCV wheel after a conflicting cv2 package uninstall..."
    sync_project --reinstall-package opencv-contrib-python
    opencv_version="$(verify_opencv)"
fi
echo "OpenCV check passed: $opencv_version"

if ! sdk_path="$(verify_sdk)"; then
    cat >&2 <<EOF

The locked '$DEPENDENCY_GROUP' group did not provide the gripper-capable A1Z
SDK. Do not start the robot. Check the uv error above, then rerun this setup.
EOF
    exit 1
fi
echo "A1Z vendor SDK check passed: $sdk_path"

if [[ "$with_lerobot" == true ]]; then
    "$python_bin" - <<'PY'
import lerobot
import pandas
import pyarrow

print(f"LeRobot check passed: {lerobot.__version__}")
PY
fi

if [[ "$sdk_only" == true ]]; then
    exit 0
fi

case "$(uname -s)" in
    Linux)
        if ! command -v sudo >/dev/null 2>&1; then
            echo "ERROR: sudo is required for Linux SocketCAN setup." >&2
            exit 1
        fi
        exec sudo "$CAN_SETUP_SCRIPT"
        ;;
    Darwin)
        if ! "$python_bin" - <<'PY' >/dev/null 2>&1
import usb.backend.libusb1

if usb.backend.libusb1.get_backend() is None:
    raise SystemExit(1)
PY
        then
            brew_bin="${A1Z_BREW_BIN:-$(command -v brew || true)}"
            if [[ -z "$brew_bin" ]]; then
                cat >&2 <<EOF
ERROR: macOS A1Z setup requires libusb, but no usable libusb backend or
Homebrew installation was found. Install Homebrew, then rerun this setup.
EOF
                exit 1
            fi
            echo "Installing the macOS libusb runtime with Homebrew..."
            "$brew_bin" install libusb
        fi

        "$python_bin" - <<'PY'
import usb.backend.libusb1
import usb.core

from dimos.hardware.manipulators.galaxea_a1z.gs_usb_bus import GsUsbMacBus

backend = usb.backend.libusb1.get_backend()
if backend is None:
    raise SystemExit("PyUSB could not load libusb after setup")
adapter = usb.core.find(idVendor=0xA8FA, idProduct=0x8598, backend=backend)
if adapter is None:
    raise SystemExit("HHS CANFD adapter a8fa:8598 was not found")
try:
    serial = getattr(adapter, "serial_number", None) or "unknown"
except usb.core.USBError:
    # The transport-level readiness retry below is authoritative; reading a
    # descriptor can race the adapter's brief macOS re-enumeration window.
    serial = "temporarily unavailable"
bus = GsUsbMacBus(listen_only=True)
try:
    print(
        "A1Z macOS USB-CAN setup passed in listen-only mode: "
        f"a8fa:8598 serial={serial}"
    )
finally:
    bus.shutdown()
PY
        echo "SocketCAN setup is not required on macOS."
        ;;
    *)
        echo "ERROR: A1Z host setup supports Linux and macOS only." >&2
        exit 1
        ;;
esac
