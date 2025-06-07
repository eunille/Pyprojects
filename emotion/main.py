import cv2
from deepface import DeepFace
from collections import Counter, deque

# Use a faster detector (mediapipe is very fast + decent accuracy)
DETECTOR_BACKEND = 'mediapipe'  # Try 'opencv' or 'mtcnn' if needed

cap = cv2.VideoCapture(0)

# Limit emotion checks to once every N frames
FRAME_SKIP = 5
frame_count = 0

# Emotion smoothing history
emotion_history = deque(maxlen=10)
last_emotion = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 480p to speed up processing
    frame = cv2.resize(frame, (480, 360))

    # Only run detection every N frames
    if frame_count % FRAME_SKIP == 0:
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            emotions = result[0]['emotion']
            dominant = result[0]['dominant_emotion']
            confidence = emotions[dominant]

            if confidence > 60:
                emotion_history.append(dominant)

                # Use the most common emotion in the recent history
                last_emotion = Counter(emotion_history).most_common(1)[0][0]

        except Exception as e:
            print("Detection error:", e)

    # Show smoothed emotion label
    cv2.putText(
        frame,
        f"{last_emotion}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Optimized Real-Time Emotion Detection", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
