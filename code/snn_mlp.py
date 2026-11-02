#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import snntorch as snn
from snntorch import surrogate
from tqdm import tqdm


ROOT = Path("/home/dinesh/Journal/Dataset/new_training_data")

CLASS_FOLDERS = [
    "Full Cable",
    "Left End",
    "Right End",
    "No Cable",
]

IMG_H, IMG_W = 60, 80

T = 1

GRID_H = 12
GRID_W = 16
N_FEATURES = GRID_H * GRID_W * 2

MODEL_TYPE = "snn"

SUBSET_SIZE = 1000000000
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEED = 42

SOBEL_PERCENTILE = 90
ALPHA = 3.0

SPIKE_PENALTY_WEIGHT = 1e-5
EARLY_EXIT_CONFIDENCE = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("/home/dinesh/Journal/snn_mlp/snn_mlp_model")

MODELS_DIR = OUT_DIR / "models"
PLOTS_DIR = OUT_DIR / "plots"
REPORTS_DIR = OUT_DIR / "reports"

for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


FEATURE_TYPE_NAME = "snn_mlp_on_off"
ARCHITECTURE_NAME = (
    f"{MODEL_TYPE.upper()} snn_mlp [T={T}, {N_FEATURES}] -> 64 -> 32 -> 4 with LIF3"
)


def image_to_spikes(img_u8):
    img_u8 = cv2.resize(
        img_u8,
        (IMG_W, IMG_H),
        interpolation=cv2.INTER_LINEAR,
    )

    img = img_u8.astype(np.float32) / 255.0

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    sobel_combined = sobel_x + sobel_y

    pos_edge = np.maximum(sobel_combined, 0)
    neg_edge = np.maximum(-sobel_combined, 0)

    pos_edge /= pos_edge.max() + 1e-8
    neg_edge /= neg_edge.max() + 1e-8

    on_mask = pos_edge > np.percentile(pos_edge, SOBEL_PERCENTILE)
    off_mask = neg_edge > np.percentile(neg_edge, SOBEL_PERCENTILE)

    spikes = np.zeros((T, 2, IMG_H, IMG_W), dtype=np.float32)

    if T == 1:
        spikes[0, 0, on_mask] = 1.0
        spikes[0, 1, off_mask] = 1.0
    else:
        on_time = (np.exp(-ALPHA * pos_edge) * (T - 1)).astype(np.int32)
        off_time = (np.exp(-ALPHA * neg_edge) * (T - 1)).astype(np.int32)

        ys, xs = np.where(on_mask)
        spikes[on_time[ys, xs], 0, ys, xs] = 1.0

        ys, xs = np.where(off_mask)
        spikes[off_time[ys, xs], 1, ys, xs] = 1.0

    return spikes


def spikes_to_snn_mlp_features(spikes):
    T_steps, _, H, W = spikes.shape

    cell_h = H // GRID_H
    cell_w = W // GRID_W

    on = spikes[:, 0].reshape(T_steps, GRID_H, cell_h, GRID_W, cell_w)
    off = spikes[:, 1].reshape(T_steps, GRID_H, cell_h, GRID_W, cell_w)

    on_density = on.mean(axis=(2, 4))
    off_density = off.mean(axis=(2, 4))

    features = np.stack([on_density, off_density], axis=-1)
    features = features.reshape(T_steps, GRID_H * GRID_W * 2)

    return features.astype(np.float32)


class CableDataset(Dataset):
    def __init__(self, root, class_folders):
        self.samples = []

        for label, cname in enumerate(class_folders):
            folder = root / cname

            if not folder.exists():
                print("Missing folder:", folder)
                continue

            count = 0
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((p, label))
                    count += 1

            print(f"{cname}: label {label}, images={count}")

        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        spikes = image_to_spikes(img)
        features = spikes_to_snn_mlp_features(spikes)

        x = torch.from_numpy(features).float()
        y = torch.tensor(label, dtype=torch.long)

        return x, y


class CachedFeatureDataset(Dataset):
    def __init__(self, base_dataset, indices, name="dataset"):
        self.X = []
        self.Y = []
        self.samples = []

        print(f"\nPrecomputing {name} features into RAM...")

        for idx in tqdm(indices, desc=f"Caching {name} features", unit="img"):
            path, label = base_dataset.samples[idx]

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise RuntimeError(f"Could not read image: {path}")

            spikes = image_to_spikes(img)
            features = spikes_to_snn_mlp_features(spikes)

            self.X.append(features)
            self.Y.append(label)
            self.samples.append((path, label))

        self.X = np.asarray(self.X, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.int64)

        print(f"{name} cached X shape:", self.X.shape)
        print(f"{name} cached Y shape:", self.Y.shape)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


def save_on_off_json(spikes, label, image_path, save_path):
    data = {
        "class_name": CLASS_FOLDERS[label],
        "image_path": str(image_path),
        "T": int(T),
        "img_h": int(IMG_H),
        "img_w": int(IMG_W),
        "grid_h": int(GRID_H),
        "grid_w": int(GRID_W),
        "on_spikes": spikes[:, 0].astype(int).tolist(),
        "off_spikes": spikes[:, 1].astype(int).tolist(),
        "total_on_spikes": int(spikes[:, 0].sum()),
        "total_off_spikes": int(spikes[:, 1].sum()),
        "total_spikes": int(spikes.sum()),
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved ON/OFF spike JSON:", save_path)


def visualize_one_image_per_class(dataset):
    seen_labels = set()

    for path, label in dataset.samples:
        if label in seen_labels:
            continue

        seen_labels.add(label)

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Could not read image:", path)
            continue

        spikes = image_to_spikes(img)

        class_name = CLASS_FOLDERS[label]
        safe_name = class_name.replace(" ", "_").lower()

        save_on_off_json(
            spikes=spikes,
            label=label,
            image_path=path,
            save_path=REPORTS_DIR / f"{safe_name}_on_off.json",
        )

        fig, axes = plt.subplots(3, T + 1, figsize=(3 * (T + 1), 9))
        fig.suptitle(
            f"Sobel ON/OFF Spike Timeline | Class: {class_name}\n{path.name}",
            fontsize=11,
        )
       

        for t in range(T):
            on_img = spikes[t, 0]
            off_img = spikes[t, 1]
            combined = on_img - off_img

            axes[0, t].imshow(on_img, cmap="gray", vmin=0, vmax=1)
            axes[0, t].set_title(f"ON t={t}")
            axes[0, t].axis("off")

            axes[1, t].imshow(off_img, cmap="gray", vmin=0, vmax=1)
            axes[1, t].set_title(f"OFF t={t}")
            axes[1, t].axis("off")

            axes[2, t].imshow(combined, cmap="bwr", vmin=-1, vmax=1)
            axes[2, t].set_title(f"ON-OFF t={t}")
            axes[2, t].axis("off")

        on_sum = spikes[:, 0].sum(axis=0)
        off_sum = spikes[:, 1].sum(axis=0)
        combined_sum = on_sum - off_sum

        axes[0, T].imshow(on_sum, cmap="gray")
        axes[0, T].set_title("ON combined")
        axes[0, T].axis("off")

        axes[1, T].imshow(off_sum, cmap="gray")
        axes[1, T].set_title("OFF combined")
        axes[1, T].axis("off")

        axes[2, T].imshow(combined_sum, cmap="bwr", vmin=-1, vmax=1)
        axes[2, T].set_title("ON-OFF combined")
        axes[2, T].axis("off")

        

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        save_path = PLOTS_DIR / f"{safe_name}_on_off_timeline.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()
        plt.close()

        print(f"Saved visualization for class '{class_name}' ->", save_path)


class CableMLP(nn.Module):
    def __init__(self, input_size=N_FEATURES * T, num_classes=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x,
        return_spike_stats=False,
        early_exit=False,
        confidence_threshold=0.90,
    ):
        x = x.reshape(x.size(0), -1)
        logits = self.net(x)

        if return_spike_stats:
            dummy_spikes = {
                "layer1": torch.zeros(x.size(0), device=x.device),
                "layer2": torch.zeros(x.size(0), device=x.device),
                "layer3": torch.zeros(x.size(0), device=x.device),
            }
            return logits, dummy_spikes, 1

        return logits


class CableSNNMLP(nn.Module):
    def __init__(
        self,
        input_size=N_FEATURES,
        hidden1=64,
        hidden2=32,
        num_classes=4,
        beta=0.9,
    ):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden2, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(
        self,
        x,
        return_spike_stats=False,
        early_exit=False,
        confidence_threshold=0.90,
    ):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        logits_sum = 0
        steps_used = 0

        layer_spikes = {
            "layer1": torch.zeros(x.size(0), device=x.device),
            "layer2": torch.zeros(x.size(0), device=x.device),
            "layer3": torch.zeros(x.size(0), device=x.device),
        }

        T_steps = x.size(1)

        for t in range(T_steps):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Train using final membrane potential as class score
            logits_sum += mem3
            steps_used += 1

            layer_spikes["layer1"] += spk1.sum(dim=1)
            layer_spikes["layer2"] += spk2.sum(dim=1)
            layer_spikes["layer3"] += spk3.sum(dim=1)

            if early_exit:
                logits_avg_now = logits_sum / steps_used
                probs = torch.softmax(logits_avg_now, dim=1)
                max_conf = probs.max(dim=1).values

                if torch.all(max_conf >= confidence_threshold):
                    break

        logits_avg = logits_sum / steps_used

        if return_spike_stats:
            return logits_avg, layer_spikes, steps_used

        return logits_avg


def plot_accuracy(train_accs, val_accs, save_path=None):
    epochs = np.arange(1, len(train_accs) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accs,
        marker="o",
        linewidth=2,
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        val_accs,
        marker="o",
        linewidth=2,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)

    plt.title(
        f"{MODEL_TYPE.upper()} snn_mlp Accuracy\n"
        f"T={T}, Grid={GRID_H}x{GRID_W}"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()


def export_fpga_weights(model, save_dir, scale=128):
    save_dir.mkdir(parents=True, exist_ok=True)

    def q_int16(arr):
        return np.round(arr * scale).astype(np.int16)

    if MODEL_TYPE == "snn":
        layers = {
            "fc1_w": model.fc1.weight.detach().cpu().numpy(),
            "fc1_b": model.fc1.bias.detach().cpu().numpy(),
            "fc2_w": model.fc2.weight.detach().cpu().numpy(),
            "fc2_b": model.fc2.bias.detach().cpu().numpy(),
            "fc3_w": model.fc3.weight.detach().cpu().numpy(),
            "fc3_b": model.fc3.bias.detach().cpu().numpy(),
        }
    else:
        layers = {
            "mlp_fc1_w": model.net[0].weight.detach().cpu().numpy(),
            "mlp_fc1_b": model.net[0].bias.detach().cpu().numpy(),
            "mlp_fc2_w": model.net[2].weight.detach().cpu().numpy(),
            "mlp_fc2_b": model.net[2].bias.detach().cpu().numpy(),
            "mlp_fc3_w": model.net[4].weight.detach().cpu().numpy(),
            "mlp_fc3_b": model.net[4].bias.detach().cpu().numpy(),
        }

    for name, arr in layers.items():
        np.save(save_dir / f"{name}_float.npy", arr)
        np.save(save_dir / f"{name}_int16.npy", q_int16(arr))

    header_path = save_dir / "fpga_weights.h"

    with open(header_path, "w") as f:
        f.write("#ifndef FPGA_WEIGHTS_H\n")
        f.write("#define FPGA_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define WEIGHT_SCALE {scale}\n")
        f.write(f"#define T_STEPS {T}\n")
        f.write(f"#define IMG_H {IMG_H}\n")
        f.write(f"#define IMG_W {IMG_W}\n")
        f.write(f"#define GRID_H {GRID_H}\n")
        f.write(f"#define GRID_W {GRID_W}\n")
        f.write(f"#define N_FEATURES {N_FEATURES}\n")
        f.write("#define HIDDEN1 64\n")
        f.write("#define HIDDEN2 32\n")
        f.write(f"#define NUM_CLASSES {len(CLASS_FOLDERS)}\n\n")

        for name, arr in layers.items():
            q = q_int16(arr)

            if q.ndim == 2:
                rows, cols = q.shape
                f.write(f"static const int16_t {name}[{rows}][{cols}] = {{\n")
                for r in range(rows):
                    row = ", ".join(str(int(v)) for v in q[r])
                    f.write(f"    {{{row}}},\n")
                f.write("};\n\n")

            elif q.ndim == 1:
                size = q.shape[0]
                values = ", ".join(str(int(v)) for v in q)
                f.write(f"static const int16_t {name}[{size}] = {{{values}}};\n\n")

        f.write("#endif\n")

    print("Saved FPGA weights to:", save_dir)
    print("Saved FPGA C header:", header_path)


def save_outputs(
    model,
    optimizer,
    train_accs,
    val_accs,
    train_losses,
    avg_val_steps_list,
):
    history = {
        "model_type": MODEL_TYPE,
        "feature_type": FEATURE_TYPE_NAME,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "n_features_per_timestep": N_FEATURES,
        "input_shape": [T, N_FEATURES],
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_ce_loss": train_losses,
        "avg_val_steps": avg_val_steps_list,
        "T": T,
        "architecture": ARCHITECTURE_NAME,
        "spike_penalty_weight": SPIKE_PENALTY_WEIGHT,
        "early_exit_confidence": EARLY_EXIT_CONFIDENCE,
        "sobel_percentile": SOBEL_PERCENTILE,
        "output_lif_layer": True,
        "loss_output": "final_lif3_membrane_potential",
        "spike_penalty_layers": ["lif1", "lif2", "lif3"],
    }

    with open(REPORTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(REPORTS_DIR / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_ce_loss",
                "train_acc",
                "val_acc",
                "avg_val_steps",
            ]
        )

        for i, (loss, ta, va, steps) in enumerate(
            zip(train_losses, train_accs, val_accs, avg_val_steps_list),
            start=1,
        ):
            writer.writerow([i, loss, ta, va, steps])

    with open(OUT_DIR / "run_info.txt", "w") as f:
        f.write(f"root={ROOT}\n")
        f.write(f"classes={CLASS_FOLDERS}\n")
        f.write(f"img_h={IMG_H}\n")
        f.write(f"img_w={IMG_W}\n")
        f.write(f"T={T}\n")
        f.write(f"grid_h={GRID_H}\n")
        f.write(f"grid_w={GRID_W}\n")
        f.write(f"n_features_per_timestep={N_FEATURES}\n")
        f.write(f"input_shape=[{T}, {N_FEATURES}]\n")
        f.write(f"model_type={MODEL_TYPE}\n")
        f.write(f"feature_type={FEATURE_TYPE_NAME}\n")
        f.write(f"subset_size={SUBSET_SIZE}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write(f"epochs={EPOCHS}\n")
        f.write(f"lr={LR}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"sobel_percentile={SOBEL_PERCENTILE}\n")
        f.write(f"architecture={ARCHITECTURE_NAME}\n")
        f.write(f"spike_penalty_weight={SPIKE_PENALTY_WEIGHT}\n")
        f.write(f"early_exit_confidence={EARLY_EXIT_CONFIDENCE}\n")
        f.write(f"device={DEVICE}\n")
        f.write("output_lif_layer=True\n")
        f.write("loss_output=final_lif3_membrane_potential\n")
        f.write("spike_penalty_layers=lif1,lif2,lif3\n")

    torch.save(
        model.state_dict(),
        MODELS_DIR / "snn_mlp_dict.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_folders": CLASS_FOLDERS,
            "model_type": MODEL_TYPE,
            "feature_type": FEATURE_TYPE_NAME,
            "img_h": IMG_H,
            "img_w": IMG_W,
            "T": T,
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "n_features_per_timestep": N_FEATURES,
            "input_shape": [T, N_FEATURES],
            "architecture": ARCHITECTURE_NAME,
            "seed": SEED,
            "sobel_percentile": SOBEL_PERCENTILE,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "train_losses": train_losses,
            "avg_val_steps": avg_val_steps_list,
            "output_lif_layer": True,
            "loss_output": "final_lif3_membrane_potential",
            "spike_penalty_layers": ["lif1", "lif2", "lif3"],
        },
        MODELS_DIR / "snn_mlp_checkpoint.pt",
    )

    export_fpga_weights(
        model=model,
        save_dir=MODELS_DIR / "fpga_export",
        scale=128,
    )

    print("\nSaved outputs to:", OUT_DIR)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_full = CableDataset(ROOT, CLASS_FOLDERS)

    visualize_one_image_per_class(dataset_full)

    if len(dataset_full) == 0:
        raise RuntimeError("No samples found.")

    subset_size = min(SUBSET_SIZE, len(dataset_full))

    all_indices = np.arange(len(dataset_full))
    all_labels = np.array([label for _, label in dataset_full.samples])

    if subset_size == len(dataset_full):
        subset_indices = all_indices
    else:
        subset_indices, _ = train_test_split(
            all_indices,
            train_size=subset_size,
            stratify=all_labels,
            random_state=SEED,
        )

    subset_labels = all_labels[subset_indices]

    train_indices, val_indices = train_test_split(
        subset_indices,
        test_size=0.30,
        stratify=subset_labels,
        random_state=SEED,
    )

    print("Using subset:", len(subset_indices), "out of", len(dataset_full))
    print("Train samples:", len(train_indices))
    print("Val samples:", len(val_indices))

    train_set = CachedFeatureDataset(dataset_full, train_indices, name="train")
    val_set = CachedFeatureDataset(dataset_full, val_indices, name="val")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if MODEL_TYPE == "snn":
        model = CableSNNMLP(
            input_size=N_FEATURES,
            num_classes=len(CLASS_FOLDERS),
        ).to(DEVICE)

    elif MODEL_TYPE == "mlp":
        model = CableMLP(
            input_size=N_FEATURES * T,
            num_classes=len(CLASS_FOLDERS),
        ).to(DEVICE)

    else:
        raise ValueError("MODEL_TYPE must be 'snn' or 'mlp'")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs = []
    train_losses = []
    avg_val_steps_list = []

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_l1_spikes = 0.0
        epoch_l2_spikes = 0.0
        epoch_l3_spikes = 0.0
        epoch_total_layer_spikes = 0.0
        epoch_samples = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits, layer_spikes, steps_used = model(
                x,
                return_spike_stats=True,
                early_exit=False,
            )

            ce_loss = criterion(logits, y)

            l1_spikes_per_image = layer_spikes["layer1"]
            l2_spikes_per_image = layer_spikes["layer2"]
            l3_spikes_per_image = layer_spikes["layer3"]

            total_layer_spikes_per_image = (
                l1_spikes_per_image
                + l2_spikes_per_image
                + l3_spikes_per_image
            )

            if MODEL_TYPE == "snn":
                spike_penalty = (
                    SPIKE_PENALTY_WEIGHT
                    * total_layer_spikes_per_image.mean()
                )
            else:
                spike_penalty = torch.tensor(0.0, device=DEVICE)

            loss = ce_loss + spike_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += ce_loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

            batch_size = x.size(0)

            epoch_l1_spikes += l1_spikes_per_image.sum().item()
            epoch_l2_spikes += l2_spikes_per_image.sum().item()
            epoch_l3_spikes += l3_spikes_per_image.sum().item()
            epoch_total_layer_spikes += total_layer_spikes_per_image.sum().item()
            epoch_samples += batch_size

            if batch_idx % 100 == 0:
                input_nonzero = (x > 0).float().sum(dim=(1, 2)).mean().item()
                input_density = 100.0 * input_nonzero / (T * N_FEATURES)

                print(
                    f"Epoch {epoch + 1:02d}/{EPOCHS} "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"ce_loss={ce_loss.item():.4f} | "
                    f"spike_penalty={spike_penalty.item():.6f} | "
                    f"loss={loss.item():.4f} | "
                    f"input_nonzero/img={input_nonzero:.1f}/{T * N_FEATURES} | "
                    f"input_density={input_density:.2f}% | "
                    f"L1/img={l1_spikes_per_image.mean().item():.1f} | "
                    f"L2/img={l2_spikes_per_image.mean().item():.1f} | "
                    f"L3/img={l3_spikes_per_image.mean().item():.1f}"
                )

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        avg_l1_spikes = epoch_l1_spikes / epoch_samples
        avg_l2_spikes = epoch_l2_spikes / epoch_samples
        avg_l3_spikes = epoch_l3_spikes / epoch_samples
        avg_total_layer_spikes = epoch_total_layer_spikes / epoch_samples

        model.eval()

        val_correct = 0
        val_total = 0
        val_steps_used = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits, _, steps_used = model(
                    x,
                    return_spike_stats=True,
                    early_exit=True,
                    confidence_threshold=EARLY_EXIT_CONFIDENCE,
                )

                pred = logits.argmax(dim=1)

                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                val_steps_used += steps_used * x.size(0)

        val_acc = val_correct / val_total
        avg_val_steps = val_steps_used / val_total

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        avg_val_steps_list.append(avg_val_steps)

        max_steps = T if MODEL_TYPE == "snn" else 1

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} DONE | "
            f"ce_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"avg_val_steps={avg_val_steps:.2f}/{max_steps} | "
            f"avg_L1/img={avg_l1_spikes:.1f} | "
            f"avg_L2/img={avg_l2_spikes:.1f} | "
            f"avg_L3/img={avg_l3_spikes:.1f} | "
            f"avg_total_layer/img={avg_total_layer_spikes:.1f}"
        )

    plot_accuracy(train_accs, val_accs, PLOTS_DIR / "accuracy_vs_epoch.png")

    save_outputs(
        model=model,
        optimizer=optimizer,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        avg_val_steps_list=avg_val_steps_list,
    )


if __name__ == "__main__":
    print("Device:", DEVICE)
    print("Model type:", MODEL_TYPE)
    print("Temporal steps T:", T)
    print("snn_mlp grid:", GRID_H, "x", GRID_W)
    print("Features per timestep:", N_FEATURES)
    print("Input shape per sample:", [T, N_FEATURES])
    print("Architecture:", ARCHITECTURE_NAME)
    print("Feature type:", FEATURE_TYPE_NAME)
    print("Training data from:", ROOT)
    print("Saving outputs to:", OUT_DIR)
    train()



