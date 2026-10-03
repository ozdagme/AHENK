"""P02-AHENK için PyTorch ve CUDA erişimini salt okunur biçimde sınar."""

try:
    import torch
except ImportError:
    print("PyTorch kurulu değil. Bu aşamada hata değildir.")
    raise SystemExit(0)

print("torch_version:", torch.__version__)
print("torch_cuda_build:", torch.version.cuda)
print("cudnn_version:", torch.backends.cudnn.version())
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"device_{index}_name:", properties.name)
        print(f"device_{index}_capability:", properties.major, properties.minor)
        print(f"device_{index}_total_vram_gib:", round(properties.total_memory / 2**30, 2))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("cuda_tensor_test:", tuple(y.shape), y.device)
    print("allocated_vram_mib:", round(torch.cuda.memory_allocated() / 2**20, 2))
else:
    print("CUDA testi çalıştırılmadı: torch.cuda.is_available() False.")