import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose model with higher accuracy settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

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

# Function to save a screenshot when the form is perfect
def save_screenshot(image, frame_number):
    filename = f"screenshot_perfect_form_frame_{frame_number}.png"
    cv2.imwrite(filename, image)
    print(f"Screenshot saved: {filename}")

# Function to analyze the sprint start in each frame
def analyze_sprint_start(image, frame_number):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Points for left leg
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Points for right leg
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Determine which leg is in front by comparing the x-coordinates of the knees
        if left_knee[0] < right_knee[0]:
            # Left leg is in front
            front_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            back_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            front_points = (left_hip, left_knee, left_ankle)
            back_points = (right_hip, right_knee, right_ankle)
        else:
            # Right leg is in front
            front_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            back_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            front_points = (right_hip, right_knee, right_ankle)
            back_points = (left_hip, left_knee, left_ankle)

        # Calculate shoulder to hand angle
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        shoulder_to_hand_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_points = (shoulder, elbow, wrist)

        # Draw the lines used for angle calculation
        def draw_angle_lines(image, points):
            cv2.line(image, tuple(np.multiply(points[0], [image.shape[1], image.shape[0]]).astype(int)),
                     tuple(np.multiply(points[1], [image.shape[1], image.shape[0]]).astype(int)), (0, 255, 0), 2)
            cv2.line(image, tuple(np.multiply(points[1], [image.shape[1], image.shape[0]]).astype(int)),
                     tuple(np.multiply(points[2], [image.shape[1], image.shape[0]]).astype(int)), (0, 255, 0), 2)

        draw_angle_lines(image, front_points)
        draw_angle_lines(image, back_points)
        draw_angle_lines(image, shoulder_points)

        # Determine if the form is perfect
        front_leg_correct_range = (85 <= front_leg_angle <= 95)
        back_leg_correct_range = (115 <= back_leg_angle <= 125)
        shoulder_to_hand_correct_range = (170 <= shoulder_to_hand_angle <= 180)

        if front_leg_correct_range and back_leg_correct_range and shoulder_to_hand_correct_range:
            print("Perfect form detected.")
            save_screenshot(image, frame_number)

        # Display angles in the top-left corner of the image
        text_position = (10, 30)
        cv2.putText(image, f'Front Leg: {int(front_leg_angle)}°', text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        text_position = (10, 70)
        cv2.putText(image, f'Back Leg: {int(back_leg_angle)}°', text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        text_position = (10, 110)
        cv2.putText(image, f'Shoulder to Hand: {int(shoulder_to_hand_angle)}°', text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image

# Load the video file
video_path = r"C:\Users\ayant\Downloads\powell_sprinter_start.mp4"
cap = cv2.VideoCapture(video_path)

# Ensure output directory exists
output_dir = "screenshots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

paused = False

# Process each frame of the video
frame_number = 0
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Analyze the sprint start in the current frame
        output_frame = analyze_sprint_start(frame, frame_number)

        # Display the result
        cv2.imshow('Sprint Start Analysis', output_frame)

    # Handle key presses
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):  # Spacebar pressed
        paused = not paused

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
