#!/usr/bin/env python3

from pathlib import Path
import struct
import time

import cv2
import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/dinesh/Journal/Dataset/new_testing_data")

SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 4000000

MAX_IMAGES_PER_FOLDER = 100

IMG_H = 60
IMG_W = 80

T_STEPS = 1

GRID_H = 12
GRID_W = 16
N_FEATURES = GRID_H * GRID_W * 2

SOBEL_PERCENTILE = 90
ALPHA = 3.0

FEATURE_SCALE = 128

MAGIC = b"\xAA\x55"

CLASS_FOLDERS = [
    "Full Cable",
    "Left End",
    "Right End",
    "No Cable",
]


# ============================================================
# TRAINING-MATCHED FEATURE EXTRACTION
# ============================================================

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

    spikes = np.zeros((T_STEPS, 2, IMG_H, IMG_W), dtype=np.float32)

    if T_STEPS == 1:
        spikes[0, 0, on_mask] = 1.0
        spikes[0, 1, off_mask] = 1.0
    else:
        on_time = (np.exp(-ALPHA * pos_edge) * (T_STEPS - 1)).astype(np.int32)
        off_time = (np.exp(-ALPHA * neg_edge) * (T_STEPS - 1)).astype(np.int32)

        ys, xs = np.where(on_mask)
        spikes[on_time[ys, xs], 0, ys, xs] = 1.0

        ys, xs = np.where(off_mask)
        spikes[off_time[ys, xs], 1, ys, xs] = 1.0

    return spikes


def spikes_to_sparse_regional_features(spikes):

    T, _, H, W = spikes.shape

    cell_h = H // GRID_H
    cell_w = W // GRID_W

    all_features = []

    for t in range(T):

        on_map = spikes[t, 0]
        off_map = spikes[t, 1]

        features = []

        for gy in range(GRID_H):
            y0 = gy * cell_h
            y1 = H if gy == GRID_H - 1 else (gy + 1) * cell_h

            for gx in range(GRID_W):
                x0 = gx * cell_w
                x1 = W if gx == GRID_W - 1 else (gx + 1) * cell_w

                on_cell = on_map[y0:y1, x0:x1]
                off_cell = off_map[y0:y1, x0:x1]

                area = float((y1 - y0) * (x1 - x0))

                features.append(float(on_cell.sum()) / area)
                features.append(float(off_cell.sum()) / area)

        all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def extract_features(image_path: Path) -> np.ndarray:

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    spikes = image_to_spikes(img)
    features_float = spikes_to_sparse_regional_features(spikes).reshape(-1)

    features_int16 = np.round(features_float * FEATURE_SCALE).astype(np.int16)

    assert features_int16.shape[0] == T_STEPS * N_FEATURES

    return features_int16


# ============================================================
# UART COMMUNICATION
# FPGA reply:
#   0xA5 <pred> <time_us uint32 little-endian> 0x5A
# ============================================================
def send_features_and_get_prediction(ser, features):

    packet = MAGIC + struct.pack(
        "<" + "h" * len(features),
        *features.tolist(),
    )

    ser.reset_input_buffer()

    ser.write(packet)
    ser.flush()

    reply = ser.read(23)

    if len(reply) != 23:
        raise RuntimeError(f"Incomplete FPGA reply: {reply}")

    if reply[0] != 0xA5:
        raise RuntimeError(f"Bad FPGA reply start: {reply}")

    if reply[22] != 0x5A:
        raise RuntimeError(f"Bad FPGA reply end: {reply}")

    pred = reply[1]
    time_us = struct.unpack("<I", reply[2:6])[0]
    logits = struct.unpack("<iiii", reply[6:22])

    time_ms = time_us / 1000.0

    return pred, time_us, time_ms, logits


# ============================================================
# LABELS
# ============================================================

def label_from_folder(folder_name: str) -> int:
    return CLASS_FOLDERS.index(folder_name)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=====================================")
    print(" FPGA UART SNN TEST")
    print(" Training-matched preprocessing")
    print("=====================================")
    print()

    print(f"Dataset root : {ROOT}")
    print(f"Serial port  : {SERIAL_PORT}")
    print(f"Baud rate    : {BAUD_RATE}")
    print(f"Images/folder: {MAX_IMAGES_PER_FOLDER}")
    print(f"T_STEPS      : {T_STEPS}")
    print(f"Grid         : {GRID_H} x {GRID_W}")
    print(f"N_FEATURES   : {N_FEATURES}")
    print(f"Scale        : {FEATURE_SCALE}")
    print()

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=5,
        write_timeout=5,
    )

    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    total = 0
    correct = 0

    confusion = np.zeros((4, 4), dtype=int)

    for class_name in CLASS_FOLDERS:

        class_dir = ROOT / class_name

        if not class_dir.exists():
            print(f"Skipping missing folder: {class_dir}")
            continue

        true_label = label_from_folder(class_name)

        image_files = sorted(
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.bmp"))
        )[:MAX_IMAGES_PER_FOLDER]

        print()
        print(f"Processing class: {class_name}")
        print(f"Images selected: {len(image_files)}")
        print()

        for image_path in image_files:

            features = extract_features(image_path)

            pred, time_us, time_ms, logits = send_features_and_get_prediction(
                ser,
                features,
            )

            total += 1
            correct += int(pred == true_label)

            if 0 <= pred < 4:
                confusion[true_label, pred] += 1

            print(
            f"{image_path.name}: "
            f"true={true_label} "
            f"pred={pred} "
            f"time={time_us} us "
            f"({time_ms:.3f} ms) "
            f"logits={logits} "
            f"{'OK' if pred == true_label else 'WRONG'}"
        )


    ser.close()

    print()
    print("=====================================")
    print(" FINAL RESULTS")
    print("=====================================")
    print()

    print(f"Total   : {total}")
    print(f"Correct : {correct}")

    if total > 0:
        print(f"Accuracy: {correct / total * 100:.2f}%")

    print()
    print("Confusion matrix")
    print("(rows=true, cols=pred)")
    print()
    print(confusion)
    print()


if __name__ == "__main__":
    main()
