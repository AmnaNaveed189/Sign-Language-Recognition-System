import cv2, pickle, numpy as np
import mediapipe as mp
from collections import deque

# ============================================================
# Load your trained model
# ============================================================
with open('models/asl_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

CLASS_NAMES = list(le.classes_)

# ============================================================
# MediaPipe setup
# ============================================================
mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,    # video mode — faster
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ============================================================
# Smoothing — average last 10 predictions
# This prevents flickering between similar letters
# ============================================================
pred_buffer   = deque(maxlen=10)
letter_buffer = []   # build words from confirmed letters
last_added    = ""
word_display  = ""

# ============================================================
# Open webcam
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("ASL Live Test started!")
print("Controls:")
print("  SPACE  = add predicted letter to word")
print("  ENTER  = finalize word / new word")
print("  BKSP   = delete last letter")
print("  C      = clear everything")
print("  Q      = quit")

CONF_THRESHOLD = 0.70   # only show prediction if confidence > 70%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally so it feels like a mirror
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_pred  = "---"
    current_conf  = 0.0
    smoothed_pred = "---"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Draw hand skeleton on frame
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        # Extract and normalize landmarks
        wrist = hand.landmark[0]
        lms = []
        for lm in hand.landmark:
            lms.extend([lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z])

        # Predict
        proba = clf.predict_proba([lms])[0]
        top_idx  = np.argmax(proba)
        current_conf = proba[top_idx]
        current_pred = CLASS_NAMES[top_idx]

        if current_conf >= CONF_THRESHOLD:
            pred_buffer.append(current_pred)

        # Smoothed prediction = most common in last 10 frames
        if pred_buffer:
            from collections import Counter
            smoothed_pred = Counter(pred_buffer).most_common(1)[0][0]

    # ============================================================
    # Draw UI overlay
    # ============================================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Main prediction display
    conf_color = (0, 200, 80) if current_conf > 0.85 else \
                 (0, 165, 255) if current_conf > 0.70 else \
                 (0, 100, 255)
    cv2.putText(frame, f"{smoothed_pred}",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.5, conf_color, 5)
    cv2.putText(frame, f"Confidence: {current_conf:.0%}",
        (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
    cv2.putText(frame, f"(raw: {current_pred})",
        (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)


    # Confidence bar
    bar_x, bar_y, bar_h = 200, 100, 14
    bar_max_w = w - 220
    bar_w = int(current_conf * bar_max_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_max_w, bar_y+bar_h),
        (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
        conf_color, -1)

    # Word being built
    word_text = "Word: " + ''.join(letter_buffer) + "_"
    cv2.putText(frame, word_text,
        (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    if word_display:
        cv2.putText(frame, f"Sentence: {word_display}",
            (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (200, 220, 255), 1)


    # Controls hint
    cv2.putText(frame,
        "SPACE=add letter | ENTER=new word | BKSP=delete | C=clear | Q=quit",
        (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imshow("ASL Live Recognition — Press Q to quit", frame)


    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and smoothed_pred not in ("---", "nothing"):
        # Add current letter to word
        letter_buffer.append(smoothed_pred)
        print(f"Added: {smoothed_pred} → {''.join(letter_buffer)}")
    elif key == 13:  # ENTER
        # Finalize word
        if letter_buffer:
            word_display += ''.join(letter_buffer) + ' '
            print(f"Word added: {''.join(letter_buffer)}")
            letter_buffer.clear()
    elif key == 8:   # BACKSPACE
        if letter_buffer:
            removed = letter_buffer.pop()
            print(f"Removed: {removed}")
    elif key == ord('c'):
        letter_buffer.clear()
        word_display = ""
        pred_buffer.clear()
        print("Cleared!")


cap.release()
cv2.destroyAllWindows()
print(f"\nFinal output: {word_display}")
