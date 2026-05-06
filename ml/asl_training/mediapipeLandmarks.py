import mediapipe as mp
import cv2, os, csv, numpy as np
from tqdm import tqdm   # pip install tqdm  (progress bar)

if not hasattr(mp, "solutions"):
    version = getattr(mp, "__version__", "unknown")
    raise RuntimeError(
        "This script requires the legacy MediaPipe Solutions API "
        "(`mp.solutions.hands`), but the installed `mediapipe` package "
        f"does not provide it. Installed version: {version}. "
        "Install a compatible release, for example:\n"
        "python -m pip install --force-reinstall mediapipe==0.10.11"
    )

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # process one image at a time
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA_DIR  = 'asl_data/asl_alphabet_train/asl_alphabet_train/'
OUTPUT_CSV = 'asl_data/landmarks.csv'

def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks relative to wrist position.
    This makes the features position-independent —
    the model learns SHAPE of hand, not WHERE in frame.
    """
    lms = []
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z

    for lm in hand_landmarks.landmark:
        lms.extend([
            lm.x - wrist_x,   # relative x
            lm.y - wrist_y,   # relative y
            lm.z - wrist_z    # relative z
        ])
    return lms

classes = sorted(os.listdir(DATA_DIR))
print(f"Processing {len(classes)} classes...")

skipped = 0
total   = 0

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row
    header = [f"x{i}" for i in range(63)] + ['label']
    writer.writerow(header)

    for label in tqdm(classes, desc="Classes"):
        class_dir = f"{DATA_DIR}/{label}/"
        images = os.listdir(class_dir)

        for img_file in tqdm(images, desc=f"  {label}", leave=False):
            img_path = f"{class_dir}/{img_file}"
            img = cv2.imread(img_path)

            if img is None:
                skipped += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                landmarks = normalize_landmarks(
                    result.multi_hand_landmarks[0])
                writer.writerow(landmarks + [label])
                total += 1
            else:
                skipped += 1   # MediaPipe couldn't detect hand

print(f"\nDone!")
print(f"Saved: {total} samples")
print(f"Skipped (no hand detected): {skipped}")
print(f"CSV saved to: {OUTPUT_CSV}")
