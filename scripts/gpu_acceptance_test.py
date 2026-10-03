"""P02-AHENK için küçük ve güvenli GPU eğitim kabul testi."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn


class ReliabilityFusion(nn.Module):
    """Üç kipi maske ve kalite değerleriyle birleştiren küçük test modeli."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.text = nn.Linear(768, hidden_size)
        self.audio = nn.Linear(128, hidden_size)
        self.price = nn.Linear(32, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        price: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        representations = torch.stack(
            [self.text(text), self.audio(audio), self.price(price)], dim=1
        )
        logits = self.gate(torch.cat([quality, mask], dim=1))
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = (representations * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(fused).squeeze(-1)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA kullanılamıyor; kabul testi durduruldu.")

    torch.manual_seed(20260927)
    device = torch.device("cuda:0")
    free_before, total = torch.cuda.mem_get_info(device)

    model = ReliabilityFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    batch_size = 64
    text = torch.randn(batch_size, 768, device=device)
    audio = torch.randn(batch_size, 128, device=device)
    price = torch.randn(batch_size, 32, device=device)
    quality = torch.rand(batch_size, 3, device=device)
    mask = torch.ones(batch_size, 3, device=device)
    mask[::4, 1] = 0
    target = torch.rand(batch_size, device=device) + 1e-4

    torch.cuda.reset_peak_memory_stats(device)

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(text, audio, price, quality, mask)
            loss = torch.mean((torch.log(prediction) - torch.log(target)) ** 2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    for _ in range(20):
        train_step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(100):
        final_loss = train_step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    free_after, _ = torch.cuda.mem_get_info(device)
    result = {
        "status": "PASS",
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_gib": round(total / 2**30, 3),
        "free_vram_before_gib": round(free_before / 2**30, 3),
        "free_vram_after_gib": round(free_after / 2**30, 3),
        "peak_test_vram_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
        "warmup_steps": 20,
        "measured_steps": 100,
        "seconds": round(elapsed, 4),
        "steps_per_second": round(100 / elapsed, 3),
        "final_loss": round(final_loss, 8),
    }

    output = Path("reports/gpu_acceptance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Rapor yazıldı: {output}")


if __name__ == "__main__":
    main()