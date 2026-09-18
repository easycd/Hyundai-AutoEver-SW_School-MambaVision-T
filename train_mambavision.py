"""Fine-tune official MambaVision-T. See README.md for CUDA setup."""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import random
import time
from pathlib import Path

from hand_data import CLASSES, collect, metrics, split_train
from result_report import roc_auc, write_report


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def exclusive_training():
    """Allow only one training process in this WSL environment."""
    import fcntl

    lock_path = Path("/tmp/hand_mambavision_training.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "알 수 없는 프로세스"
            raise RuntimeError(f"다른 학습이 이미 실행 중입니다 ({owner}).") from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"PID {os.getpid()}")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class HandDataset:
    def __init__(self, root, rows, size, mean, std, augment=False, cache_dir=None):
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        from torchvision import transforms
        register_heif_opener()
        self.root, self.rows, self.size = Path(root), rows, size
        self.Image, self.ImageOps = Image, ImageOps
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parent / ".image_cache" / str(size))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        operations = []
        if augment:
            operations = [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
        self.transform = transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.rows)

    def prepared_image(self, index):
        """Return the exact letterboxed image used by the model, using disk cache."""
        row = self.rows[index]
        path = self.root / row["path"]
        try:
            stat = path.stat()
            cache_key = hashlib.sha1(
                f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{self.size}".encode("utf-8")
            ).hexdigest()
            cached = self.cache_dir / f"{cache_key}.png"
            if cached.exists():
                with self.Image.open(cached) as prepared:
                    image = prepared.convert("RGB")
            else:
                with self.Image.open(path) as original:
                    image = self.ImageOps.exif_transpose(original).convert("RGB")
                    # Letterbox: preserve ALL image content and aspect ratio.
                    image = self.ImageOps.pad(image, (self.size, self.size),
                                             method=self.Image.Resampling.BICUBIC, color=(124, 116, 104))
                temporary = self.cache_dir / f"{cache_key}.{os.getpid()}.tmp.png"
                image.save(temporary, format="PNG", compress_level=1)
                try:
                    temporary.replace(cached)
                except FileExistsError:
                    temporary.unlink(missing_ok=True)
            return image
        except Exception as error:
            raise RuntimeError(f"Cannot decode image: {path}: {error}") from error

    def __getitem__(self, index):
        image = self.prepared_image(index)
        return self.transform(image), self.rows[index]["label"]


def run_epoch(model, loader, criterion, torch, optimizer=None, head_only=False):
    training = optimizer is not None
    model.train(training)
    if training:
        # Small dataset: do not overwrite pretrained BatchNorm running statistics.
        if head_only:
            model.eval()
            model.head.train()
        else:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
    targets, predictions, probabilities = [], [], []
    total_loss = 0.0
    forward_seconds = 0.0
    warmed_up = False
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if not training:
                if not warmed_up:
                    for _ in range(3):
                        model(images)
                    warmed_up = True
                torch.cuda.synchronize()
                started = time.perf_counter()
            logits = model(images)
            if not training:
                torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - started
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; training stopped")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
            probabilities.extend(logits.softmax(1)[:, 1].detach().tolist())
    result = metrics(targets, predictions)
    result["loss"] = total_loss / len(targets)
    result["roc_auc"] = roc_auc(targets, probabilities)
    result["inference_ms_per_image"] = None if training else forward_seconds * 1000 / len(targets)
    return result, predictions, probabilities


def train_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "hand_img")
    parser.add_argument("--output", type=Path, default=Path("runs/mambavision_t"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4, help="Parallel image decoding workers")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-data", action="store_true", help="Inventory/split only; no image decoding or GPU required")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a trusted checkpoint on test only")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size, args.patience) <= 0 or args.head_epochs < 0 or args.workers < 0:
        parser.error("epochs, batch-size, image-size, patience must be positive; head-epochs >= 0")
    if args.lr <= 0 or args.head_lr <= 0:
        parser.error("learning rates must be positive")
    if not args.evaluate:
        train_rows, val_rows = split_train(collect(args.data_dir, "train"), args.val_fraction, args.seed)
        for name, rows in [("train", train_rows), ("validation", val_rows)]:
            print(name, len(rows), {k: sum(r["label"] == v for r in rows) for k, v in CLASSES.items()})
        if args.check_data:
            return
    elif args.check_data:
        parser.error("--check-data cannot be combined with --evaluate")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required by Mamba selective_scan. Use Linux/WSL2 with CUDA PyTorch; see README.md")
    from mambavision import create_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = None
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        if checkpoint["class_to_idx"] != CLASSES:
            raise ValueError("Checkpoint label mapping mismatch")
        args.image_size = checkpoint["image_size"]
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"Use a new --output directory: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / ".cache" / "hand_mambavision"
    cache.mkdir(parents=True, exist_ok=True)
    # Load the original 1000-class weights BEFORE replacing the classifier.
    # NVIDIA's checkpoint includes training arguments as argparse.Namespace.
    # Keep weights_only loading enabled with a narrowly scoped allowlist.
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = create_model("mamba_vision_T", pretrained=checkpoint is None,
                             model_path=str(cache / "mambavision_tiny_1k.pth.tar"))
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model.num_classes = 2
    mean = list(model.default_cfg["mean"]) if checkpoint is None else checkpoint["mean"]
    std = list(model.default_cfg["std"]) if checkpoint is None else checkpoint["std"]
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda()

    def loader(rows, augment=False):
        dataset = HandDataset(args.data_dir, rows, args.image_size, mean, std, augment)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=augment,
                                           num_workers=args.workers, pin_memory=True,
                                           persistent_workers=args.workers > 0,
                                           generator=torch.Generator().manual_seed(args.seed))

    criterion = torch.nn.CrossEntropyLoss()
    if checkpoint:
        rows = collect(args.data_dir, "test")
        result, predictions, probabilities = run_epoch(model, loader(rows), criterion, torch)
        destination = args.evaluate.parent
        write_json(destination / "test_metrics.json", result)
        summary_path = destination / "training_summary.json"
        train_seconds = json.loads(summary_path.read_text(encoding="utf-8"))["train_seconds"] if summary_path.exists() else None
        write_report(destination / "test_results.txt", "test", result, rows, predictions,
                     train_seconds, sum(p.numel() for p in model.parameters()), args.batch_size,
                     torch.cuda.get_device_name(0))
        with (destination / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "true_label", "predicted_label", "violation_probability"])
            writer.writerows((r["path"], r["label"], p, s) for r, p, s in zip(rows, predictions, probabilities))
        print(json.dumps(result, indent=2))
        return

    write_json(args.output / "split.json", {"class_to_idx": CLASSES, "train": train_rows, "validation": val_rows})
    write_json(args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    import importlib.metadata
    write_json(args.output / "versions.json", {name: importlib.metadata.version(name) for name in
               ["torch", "torchvision", "mambavision", "timm", "mamba-ssm", "pillow-heif"]})
    train_loader, val_loader = loader(train_rows, True), loader(val_rows)
    print(f"Image cache: {(Path(__file__).resolve().parent / '.image_cache' / str(args.image_size))}")
    print(f"DataLoader workers: {args.workers} (first epoch creates cache; later epochs reuse it)", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("head.")], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, stale, history = (-1.0, float("-inf")), 0, []
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        head_only = epoch < args.head_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not head_only or name.startswith("head."))
        training, _, _ = run_epoch(model, train_loader, criterion, torch, optimizer, head_only)
        validation, val_predictions, _ = run_epoch(model, val_loader, criterion, torch)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train": training, "validation": validation})
        write_json(args.output / "history.json", history)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={training['loss']:.4f} "
              f"val_loss={validation['loss']:.4f} val_F1={validation['violation_f1']:.4f}", flush=True)
        score = (validation["violation_f1"], -validation["loss"])
        if score > best:
            best, stale = score, 0
            best_validation, best_predictions, best_epoch = validation, val_predictions, epoch + 1
            torch.save({"state_dict": model.state_dict(), "model_name": "mamba_vision_T",
                        "class_to_idx": CLASSES, "image_size": args.image_size,
                        "mean": mean, "std": std, "epoch": epoch + 1,
                        "validation": validation}, args.output / "best.pt")
        elif not head_only:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - training_started
    write_json(args.output / "training_summary.json", {"train_seconds": train_seconds,
               "epochs_completed": len(history), "best_epoch": best_epoch})
    write_report(args.output / "validation_results.txt", f"validation / best epoch={best_epoch}",
                 best_validation, val_rows, best_predictions, train_seconds,
                 sum(p.numel() for p in model.parameters()), args.batch_size, torch.cuda.get_device_name(0))
    print(f"Saved best model: {args.output / 'best.pt'}; test was not used.")
    print(f"Saved report: {args.output / 'validation_results.txt'}")


def main():
    with exclusive_training():
        train_main()


if __name__ == "__main__":
    main()
