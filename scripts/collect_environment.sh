#!/usr/bin/env bash

# P02-AHENK: WSL2/Conda/GPU ortam envanteri.
# Yalnızca bilgi okur; sistem ayarlarını değiştirmez.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ahenk-matplotlib"

section() {
  printf '\n===== %s =====\n' "$1"
}

run_if_available() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1
  else
    printf '%s bulunamadı.\n' "$command_name"
  fi
}

section "ZAMAN"
date --iso-8601=seconds

section "LINUX / WSL"
uname -a
if [[ -r /etc/os-release ]]; then
  cat /etc/os-release
fi
printf '\n/proc/version:\n'
cat /proc/version

section "CPU"
run_if_available lscpu

section "BELLEK"
run_if_available free -h
printf '\nWSL tarafından görülen toplam bellek:\n'
awk '/MemTotal|SwapTotal/ {print}' /proc/meminfo

section "DISK"
df -h .

section "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi
else
  printf 'nvidia-smi bulunamadı.\n'
fi

section "CONDA"
if command -v conda >/dev/null 2>&1; then
  conda --version
  conda info
  conda env list
else
  printf 'conda bulunamadı.\n'
fi

section "PYTHON"
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
import os
import platform
import sys

print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("machine:", platform.machine())
print("cpu_count:", os.cpu_count())
print("prefix:", sys.prefix)
PY
else
  printf 'python bulunamadı.\n'
fi

section "PYTORCH / CUDA"
if command -v python >/dev/null 2>&1; then
  python "$SCRIPT_DIR/verify_pytorch.py"
fi

section "TEMEL PAKETLER"
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
from importlib import import_module

packages = [
    "numpy", "pandas", "scipy", "sklearn", "pyarrow",
    "matplotlib", "seaborn", "yaml", "pydantic", "psutil",
    "torch", "transformers", "datasets",
]

for name in packages:
    try:
        module = import_module(name)
        version = getattr(module, "__version__", "sürüm alanı yok")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: KURULU DEĞİL ({type(exc).__name__})")
PY
fi

section "BİTTİ"
printf 'Bu raporda ortam değişkenleri ve kullanıcı dosyaları yazdırılmamıştır.\n'