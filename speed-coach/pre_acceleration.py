import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to analyze the sprint start in each frame
def analyze_sprint_start(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Extract relevant points
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Back leg points (assuming left side for both legs)
        back_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        back_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles
        front_leg_angle = calculate_angle(hip, knee, ankle)
        back_leg_angle = calculate_angle(hip, back_knee, back_ankle)
        shoulder_to_hand_angle = calculate_angle(shoulder, elbow, wrist)

        # Determine if the angles are within the correct range
        front_leg_correct_range = (85 <= front_leg_angle <= 95)
        back_leg_correct_range = (115 <= back_leg_angle <= 125)
        shoulder_to_hand_correct_range = (170 <= shoulder_to_hand_angle <= 180)

        if front_leg_correct_range:
            print("Front leg angle is correct.")
        else:
            print(f"Front leg angle is incorrect: {front_leg_angle}°. It should be around 90°.")

        if back_leg_correct_range:
            print("Back leg angle is correct.")
        else:
            print(f"Back leg angle is incorrect: {back_leg_angle}°. It should be around 120°.")

        if shoulder_to_hand_correct_range:
            print("Shoulder to hand angle is correct.")
        else:
            print(
                f"Shoulder to hand angle is incorrect: {shoulder_to_hand_angle}°. It should be between 170° and 180°.")

        # Draw angles on the image
        cv2.putText(image, f'{int(front_leg_angle)}°',
                    tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'{int(back_leg_angle)}°',
                    tuple(np.multiply(back_knee, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'{int(shoulder_to_hand_angle)}°',
                    tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image


# Load the video file
video_path = r"C:\Users\ayant\Downloads\powell_sprinter_start.mp4"
cap = cv2.VideoCapture(video_path)

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the sprint start in the current frame
    output_frame = analyze_sprint_start(frame)

    # Display the result
    cv2.imshow('Sprint Start Analysis', output_frame)

    # Press 'q' to exit the video before it ends
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
