import cv2
import time
import csv
import os
import sys
try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore[import]
    from mediapipe.tasks.python import vision  # type: ignore[import]
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore[import]
except Exception:
    print("Error: mediapipe package not found. Install with: pip install mediapipe")
    sys.exit(1)

# ── Setup ────────────────────────────────────────────────
hand_opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

DATA_FILE    = "gesture_data.csv"
COLLECT_SECS = 15      # seconds to collect per gesture
FPS_TARGET   = 20

# Gestures to collect — add/remove as you like
GESTURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
            "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "open_palm", "fist", "thumbs_up"]

# ── CSV header ───────────────────────────────────────────
def init_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            # 21 landmarks x (x, y, z) = 63 features + label
            header = []
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            header.append("label")
            writer.writerow(header)
        print(f"Created {DATA_FILE}")
    else:
        print(f"Appending to existing {DATA_FILE}")


def extract_landmarks(result):
    """Flatten 21 landmarks into a list of 63 values."""
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    row = []
    for point in lm:
        row += [round(point.x, 6), round(point.y, 6), round(point.z, 6)]
    return row


def collect_gesture(cap, gesture_name, writer):
    print(f"\n>>> GET READY for '{gesture_name}' — hold the gesture steady!")
    
    # 3 second countdown
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 100, 200), 4)
            cv2.putText(frame, f"NEXT: '{gesture_name}'", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Starting in {i}...", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, "Get your hand in position!", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.imshow("Data Collection | Q=Skip gesture  ESC=Stop all", frame)
            cv2.waitKey(1000)

    # Collect for COLLECT_SECS seconds
    start     = time.time()
    count     = 0
    skipped   = False

    while time.time() - start < COLLECT_SECS:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)

        result = hand_detector.detect_for_video(mp_image, timestamp)
        row    = extract_landmarks(result)

        elapsed = time.time() - start
        remaining = COLLECT_SECS - elapsed

        # Draw hand dots if detected
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for lm_point in result.hand_landmarks[0]:
                x, y = int(lm_point.x * w), int(lm_point.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if row:
            writer.writerow(row + [gesture_name])
            count += 1
            status_color = (0, 255, 0)
            status_text  = f"RECORDING '{gesture_name}'  — {count} samples"
        else:
            status_color = (0, 80, 255)
            status_text  = "NO HAND DETECTED — show your hand!"

        # UI
        cv2.rectangle(frame, (0, 0), (640, 480), (0, 200, 0), 3)
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Time left: {remaining:.1f}s", (430, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(frame, "Q=skip  ESC=stop", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        cv2.imshow("Data Collection | Q=Skip gesture  ESC=Stop all", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            print("Stopped by user.")
            return "stop"
        elif key == ord('q'):  # skip this gesture
            print(f"Skipped '{gesture_name}'")
            skipped = True
            break

    if not skipped:
        print(f"Collected {count} samples for '{gesture_name}'")
    return "continue"


def main():
    init_csv()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n========================================")
    print("  GESTURE DATA COLLECTION")
    print("========================================")
    print(f"Gestures to collect: {GESTURES}")
    print(f"Each gesture: {COLLECT_SECS} seconds")
    print("Q = skip current gesture")
    print("ESC = stop everything")
    print("========================================\n")

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        for gesture in GESTURES:
            action = collect_gesture(cap, gesture, writer)
            if action == "stop":
                break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()

    print(f"\nDone! Data saved to {DATA_FILE}")
    print("Now run: python train_model.py")


if __name__ == "__main__":
    main()