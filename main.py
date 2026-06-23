import cv2
import time
import os
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from collections import deque, Counter

# ── Optional spell checker ───────────────────────────────
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
except ImportError:
    spell = None
    print("pyspellchecker not installed — autocorrect disabled")

def autocorrect(word):
    if len(word) <= 1:
        return word
    if spell:
        corrected = spell.correction(word.lower())
        return corrected.upper() if corrected else word.upper()
    return word.upper()

# ── MediaPipe hand detector ──────────────────────────────
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

hand_opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

# ── Load trained gesture model ───────────────────────────
MODEL_FILE = "gesture_model.pkl"
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        gesture_model = pickle.load(f)
    print("Gesture model loaded!")
else:
    gesture_model = None
    print("WARNING: No model found. Run train_model.py first.")

# ── Custom gesture mappings ──────────────────────────────
CUSTOM_GESTURES = {
    "thumbs_up" : "GREAT",
    "open_palm" : "SPACE",
    "fist"      : "BACK",
}
SPECIAL = {"SPACE", "BACK", "CLEAR"}

# ── App constants ────────────────────────────────────────
WINDOW_NAME       = "Sign Language Pipeline | Q=Quit"
CONFIDENCE_THRESH = 30.0
HOLD_TIME         = 1.2

# ── Word builder state ───────────────────────────────────
prediction_buffer = deque(maxlen=12)
word_buffer       = []
sentence          = []
subtitle_history  = []
last_letter       = ""
hold_start        = None


# ════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════════════

def draw_hand_landmarks(frame, detection_result):
    if not detection_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        for connection in HAND_CONNECTIONS:
            s = hand_landmarks[connection.start]
            e = hand_landmarks[connection.end]
            cv2.line(frame,
                     (int(s.x*w), int(s.y*h)),
                     (int(e.x*w), int(e.y*h)),
                     (0, 180, 255), 2)
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (255,255,255), 1)


def draw_subtitle(frame, subtitle_history):
    if not subtitle_history:
        return
    h, w = frame.shape[:2]
    text = subtitle_history[-1]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = (w - tw) // 2
    y = h - 100
    cv2.rectangle(frame, (x-10, y-th-8), (x+tw+10, y+8), (0,0,0), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


def draw_ui(frame, fps):
    h, w = frame.shape[:2]
    # FPS top left
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    # Mode label top right
    cv2.putText(frame, "MODE: SIGN LANGUAGE", (w-330, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,180,0), 2)
    # Bottom controls bar
    cv2.rectangle(frame, (0, h-44), (w, h), (30,30,30), -1)
    cv2.putText(frame,
                "open_palm=SPACE   fist=BACK   thumbs_up=GREAT   Q=Quit",
                (10, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170,170,170), 1)
    return frame


# ════════════════════════════════════════════════════════
#  SIGN LANGUAGE PROCESSING
# ════════════════════════════════════════════════════════

def process_sign(frame, mp_image, timestamp):
    global last_letter, hold_start, word_buffer, sentence, subtitle_history

    result = hand_detector.detect_for_video(mp_image, timestamp)
    draw_hand_landmarks(frame, result)

    h, w = frame.shape[:2]

    if result.hand_landmarks:
        # ── Extract 63 landmark values ───────────────────
        lm  = result.hand_landmarks[0]
        row = []
        for point in lm:
            row += [point.x, point.y, point.z]

        if gesture_model is not None:
            # ── Predict ──────────────────────────────────
            raw_pred   = gesture_model.predict([row])[0]
            confidence = max(gesture_model.predict_proba([row])[0]) * 100
            pred       = CUSTOM_GESTURES.get(raw_pred, raw_pred)

            # ── Smooth across 12 frames ───────────────────
            prediction_buffer.append(pred)
            stable_pred = Counter(prediction_buffer).most_common(1)[0][0]

            # ── Confidence filter ─────────────────────────
            if confidence < CONFIDENCE_THRESH:
                display_pred = "?"
                color        = (100, 100, 100)
            else:
                display_pred = stable_pred
                color        = (0, 255, 100)

            # ── Hold timer ────────────────────────────────
            if confidence >= CONFIDENCE_THRESH:
                if stable_pred == last_letter:
                    if hold_start is None:
                        hold_start = time.time()
                    held     = time.time() - hold_start
                    progress = min(held / HOLD_TIME, 1.0)

                    # Progress bar
                    cv2.rectangle(frame, (20, 168), (220, 186), (60,60,60), -1)
                    cv2.rectangle(frame, (20, 168),
                                  (20 + int(progress*200), 186), (0,255,180), -1)
                    cv2.putText(frame, "Hold to add...", (230, 182),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

                    # ── Trigger action when hold complete ─
                    if held >= HOLD_TIME:
                        hold_start = None

                        if stable_pred == "SPACE":
                            if word_buffer:
                                raw_word  = "".join(word_buffer)
                                corrected = autocorrect(raw_word)
                                sentence.append(corrected)
                                word_buffer = []

                        elif stable_pred == "BACK":
                            if word_buffer:
                                word_buffer.pop()
                            elif sentence:
                                word_buffer = list(sentence.pop())

                        elif stable_pred == "CLEAR":
                            word_buffer      = []
                            sentence         = []
                            subtitle_history = []

                        elif stable_pred not in SPECIAL:
                            word_buffer.append(stable_pred)
                else:
                    last_letter = stable_pred
                    hold_start  = None
            else:
                hold_start = None

            # ── Big letter display ────────────────────────
            cv2.rectangle(frame, (0, 55), (w, 162), (20,20,20), -1)
            cv2.putText(frame, display_pred, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 7)

            # ── Confidence bar ────────────────────────────
            bar_fill = int((confidence / 100) * 280)
            bar_col  = color if confidence >= CONFIDENCE_THRESH else (80,80,80)
            cv2.rectangle(frame, (220, 80), (500, 106), (60,60,60), -1)
            cv2.rectangle(frame, (220, 80), (220+bar_fill, 106), bar_col, -1)
            cv2.putText(frame, f"{confidence:.1f}%", (508, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_col, 2)
            cv2.putText(frame, f"confidence (need >{CONFIDENCE_THRESH:.0f}%)",
                        (220, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130,130,130), 1)

    else:
        # No hand detected
        prediction_buffer.clear()
        hold_start = None
        cv2.putText(frame, "Show your hand to the camera",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

    # ── Word + sentence display bar ───────────────────────
    current_word = "".join(word_buffer)
    all_words    = sentence + ([current_word] if current_word else [])
    full_text    = " ".join(all_words)

    if full_text and (not subtitle_history or subtitle_history[-1] != full_text):
        subtitle_history = subtitle_history[-2:] + [full_text]

    display_text = full_text
    if len(display_text) > 38:
        display_text = "..." + display_text[-35:]

    cv2.rectangle(frame, (0, h-88), (w, h-44), (25,25,55), -1)
    cv2.putText(frame, "TEXT:", (10, h-62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,180), 1)
    cv2.putText(frame, display_text if display_text else "_",
                (75, h-58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,180), 2)

    draw_subtitle(frame, subtitle_history)

    return frame


# ════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

    print("=" * 45)
    print("  Sign Language Pipeline — READY")
    print("  open_palm = SPACE (complete word)")
    print("  fist      = BACK  (delete letter)")
    print("  thumbs_up = GREAT (custom)")
    print("  Q         = Quit")
    print("=" * 45)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = process_sign(frame, mp_image, timestamp)
        frame = draw_ui(frame, fps)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    print("Bye!")


if __name__ == "__main__":
    main()