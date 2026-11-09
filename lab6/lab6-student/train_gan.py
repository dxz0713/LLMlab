"""Train a DCGAN on Fashion-MNIST and save reproducible artifacts.

Examples:
    uv run python train_gan.py --epochs 10
    uv run python train_gan.py --smoke --epochs 1 --max-batches 2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from src.data import fashion_mnist_loader
from src.metrics import batch_diversity, interpolate_latents
from src.models import Discriminator, Generator, weights_init
from src.training import sample_noise, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DCGAN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--real-label", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use FakeData; no download required")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(args: argparse.Namespace) -> DataLoader:
    if not args.smoke:
        return fashion_mnist_loader(args.data_root, args.batch_size, args.num_workers)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.FakeData(
        size=max(args.batch_size * 4, 256), image_size=(1, 28, 28), num_classes=10, transform=transform
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def save_curves(history: list[dict[str, float]], out_dir: Path) -> None:
    matplotlib.use("Agg")
    epochs = range(1, len(history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["d_loss"] for row in history], marker="o", label="D loss")
    plt.plot(epochs, [row["g_loss"] for row in history], marker="s", label="G loss")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.title("DCGAN training losses")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()


@torch.no_grad()
def save_epoch_samples(generator: Generator, fixed_z: torch.Tensor, path: Path) -> torch.Tensor:
    generator.eval()
    images = generator(fixed_z)
    utils.save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))
    return images


@torch.no_grad()
def save_interpolation(generator: Generator, z_dim: int, device: torch.device, path: Path) -> None:
    generator.eval()
    endpoints = sample_noise(2, z_dim, device)
    z_line = interpolate_latents(endpoints[0], endpoints[1], steps=10)
    images = generator(z_line)
    utils.save_image(images, path, nrow=10, normalize=True, value_range=(-1, 1))


def main() -> tuple[Generator, Discriminator, list[dict[str, float]]]:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 < args.real_label <= 1.0:
        raise ValueError("real-label must lie in (0, 1]")

    seed_everything(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}; outputs={out_dir}")

    loader = make_loader(args)
    generator = Generator(args.z_dim, args.base_channels).to(device)
    discriminator = Discriminator(args.base_channels).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_z = sample_noise(64, args.z_dim, device)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            generator,
            discriminator,
            loader,
            optimizer_g,
            optimizer_d,
            args.z_dim,
            device,
            real_label=args.real_label,
            max_batches=args.max_batches,
        )
        generated = save_epoch_samples(generator, fixed_z, out_dir / f"samples_epoch_{epoch:03d}.png")
        stats["diversity"] = float(batch_diversity(generated).item())
        history.append(stats)
        print(
            f"epoch {epoch:03d}/{args.epochs}: D={stats['d_loss']:.4f}, "
            f"G={stats['g_loss']:.4f}, D(real)={stats['d_real']:.3f}, "
            f"D(fake)={stats['d_fake']:.3f}, diversity={stats['diversity']:.4f}"
        )

    save_curves(history, out_dir)
    save_interpolation(generator, args.z_dim, device, out_dir / "latent_interpolation.png")
    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "args": vars(args)},
        out_dir / "checkpoint.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("saved loss_curve.png, samples, latent_interpolation.png, checkpoint.pt, history.json")
    return generator, discriminator, history


if __name__ == "__main__":
    main()

