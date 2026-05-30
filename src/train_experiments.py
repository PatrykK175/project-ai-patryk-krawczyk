import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class CNNClassifier(nn.Module):
    def __init__(self, depth: int = 2, base_channels: int = 32, activation: str = "relu"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: List[nn.Module] = []
        in_channels = 1
        out_channels = base_channels

        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(self._activation(activation))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*layers)
        flattened_dim = self._infer_flattened_dim()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            self._activation(activation),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    @staticmethod
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        if name == "elu":
            return nn.ELU()
        raise ValueError(f"Unsupported activation: {name}")

    def _infer_flattened_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            y = self.features(x)
            return int(np.prod(y.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def build_dataloaders(
    batch_size: int,
    data_dir: Path,
    num_workers: int,
    quick_mode: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_full = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_size = int(0.9167 * len(train_full))  # 55k
    val_size = len(train_full) - train_size     # 5k
    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    if quick_mode:
        train_set = Subset(train_set, range(min(2048, len(train_set))))
        val_set = Subset(val_set, range(min(512, len(val_set))))
        test_set = Subset(test_set, range(min(512, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Wsteczna propagacja + aktualizacja parametrów.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        batches += 1

    return total_loss / batches, total_acc / batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, labels)
            batches += 1

    return total_loss / batches, total_acc / batches


def save_history(history: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: List[Dict[str, float]], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()


def save_misclassified_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_items: int = 25,
) -> None:
    model.eval()
    images_out = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            mask = preds != labels
            if mask.any():
                wrong_images = images[mask].cpu()
                wrong_true = labels[mask].cpu()
                wrong_pred = preds[mask].cpu()

                for img, t, p in zip(wrong_images, wrong_true, wrong_pred):
                    images_out.append(img)
                    true_labels.append(int(t))
                    pred_labels.append(int(p))
                    if len(images_out) >= max_items:
                        break

            if len(images_out) >= max_items:
                break

    if not images_out:
        return

    cols = 5
    rows = int(np.ceil(len(images_out) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, (img, t, p) in enumerate(zip(images_out, true_labels, pred_labels), start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(img.squeeze(0), cmap="gray")
        plt.title(f"T:{t} P:{p}")
        plt.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_maps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_maps: int = 16,
) -> None:
    model.eval()
    first_batch = next(iter(loader))[0][:1].to(device)

    activations: List[torch.Tensor] = []

    def hook_fn(_, __, output):
        activations.append(output.detach().cpu())

    first_conv = None
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break

    if first_conv is None:
        return

    handle = first_conv.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(first_batch)
    handle.remove()

    if not activations:
        return

    fmap = activations[0][0]
    n_maps = min(max_maps, fmap.shape[0])
    cols = 4
    rows = int(np.ceil(n_maps / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[i], cmap="viridis")
        plt.axis("off")
        plt.title(f"Map {i}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_optimizer(name: str, params, lr: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_experiment(
    config: Dict,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    exp_dir = args.output_dir / config["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=config["batch_size"],
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        quick_mode=args.quick,
        seed=args.seed,
    )

    model = CNNClassifier(
        depth=config["depth"],
        base_channels=config["base_channels"],
        activation=config["activation"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["lr"])

    history: List[Dict[str, float]] = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), exp_dir / "best_model.pt")

        print(
            f"[{config['name']}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    save_history(history, exp_dir / "history.csv")
    plot_history(history, exp_dir, config["name"])
    save_misclassified_examples(model, test_loader, device, exp_dir / "misclassified.png")
    save_feature_maps(model, test_loader, device, exp_dir / "feature_maps.png")

    summary = {
        "name": config["name"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "depth": config["depth"],
        "activation": config["activation"],
        "optimizer": config["optimizer"],
        "best_val_acc": round(best_val_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
    }

    with (exp_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def save_global_summary(results: List[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_experiments_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "lr",
                "batch_size",
                "depth",
                "activation",
                "optimizer",
                "best_val_acc",
                "test_loss",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST CNN experiments in PyTorch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Use tiny subsets for smoke tests")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiments = [
        {
            "name": "exp1_adam_lr1e3_depth2_relu_bs128",
            "lr": 1e-3,
            "batch_size": 128,
            "depth": 2,
            "activation": "relu",
            "optimizer": "adam",
            "base_channels": 32,
        },
        {
            "name": "exp2_adam_lr5e4_depth3_relu_bs128",
            "lr": 5e-4,
            "batch_size": 128,
            "depth": 3,
            "activation": "relu",
            "optimizer": "adam",
            "base_channels": 32,
        },
        {
            "name": "exp3_sgd_lr1e2_depth2_lrelu_bs64",
            "lr": 1e-2,
            "batch_size": 64,
            "depth": 2,
            "activation": "leaky_relu",
            "optimizer": "sgd",
            "base_channels": 32,
        },
    ]

    results = []
    for config in experiments:
        print(f"\nRunning experiment: {config['name']}")
        summary = run_experiment(config, args, device)
        results.append(summary)

    save_global_summary(results, args.output_dir)
    print("\nDone. Results saved in:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
