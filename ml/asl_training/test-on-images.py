import mediapipe as mp
import cv2, pickle, numpy as np

# Load everything
with open('models/asl_classifier.pkl','rb') as f: clf = pickle.load(f)
with open('models/label_encoder.pkl','rb') as f: le = pickle.load(f)

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

def predict_image(image_path):
    """Predict ASL letter from a new image"""
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, 0, "No hand detected"

    lms = []
    hand = result.multi_hand_landmarks[0]
    wrist = hand.landmark[0]
    for lm in hand.landmark:
        lms.extend([lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z])

    proba = clf.predict_proba([lms])[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.classes_[i], proba[i]) for i in top3_idx]
    return top3[0][0], top3[0][1], top3

# Test on new images
test_new = [
    "test_images/test_image_1.jpg",
    "test_images/test_image_2.jpg"
]

print("Testing on new unseen images:")
print("=" * 50)
for path in test_new:
    pred, conf, top3 = predict_image(path)
    print(f"Image: {path}")
    if pred:
        print(f"  Prediction: {pred} ({conf:.1%})")
        print(f"  Top 3: {[(l, f'{c:.0%}') for l,c in top3]}")
    else:
        print(f"  ERROR: No hand detected in image")
    print()

