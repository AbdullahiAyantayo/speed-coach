import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize video capture
video_file = r"C:\Users\ayant\Downloads\noah_lyles_start.mp4"  # Replace with your actual video file path
cap = cv2.VideoCapture(video_file)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize variables
prev_time = None
prev_centers = {}


# Function to calculate speed
def calculate_speed(current_center, prev_center, time_diff):
    distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
    speed = (distance / time_diff) * 0.0682  # Convert pixels/sec to mph (approximation)
    return speed


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = pose.process(rgb_frame)

    current_centers = {}

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx == mp_pose.PoseLandmark.NOSE.value:  # Use the nose as the tracking point
                center = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                current_centers[idx] = center

                # Draw a bounding box and center point
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                if idx in prev_centers:
                    # Calculate and display speed
                    time_diff = time.time() - prev_time
                    speed = calculate_speed(center, prev_centers[idx], time_diff)
                    cv2.putText(frame, f"{speed:.2f} mph", (center[0] + 10, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update previous values
    prev_centers = current_centers
    prev_time = time.time()

    # Display the resulting frame
    cv2.imshow('Human Speed Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
