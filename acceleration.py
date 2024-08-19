import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def provide_feedback(body_angle, thigh_angle, knee_foot_angle, arm_angle):
    """Provide feedback based on calculated angles."""
    feedback = []

    if 44 <= body_angle <= 46:
        feedback.append("Body angle: Ideal for acceleration!")
    else:
        feedback.append(f"Body angle: {int(body_angle)}°. Aim for 45° for optimal acceleration.")

    if 89 <= thigh_angle <= 91:
        feedback.append("Thigh angle: Perfect drive!")
    else:
        feedback.append(f"Thigh angle: {int(thigh_angle)}°. Aim for 90°.")

    if 89 <= knee_foot_angle <= 91:
        feedback.append("Knee-foot angle: Perfect for stride!")
    else:
        feedback.append(f"Knee-foot angle: {int(knee_foot_angle)}°. Aim for 90°.")

    if arm_angle < 90:
        feedback.append("Arm position: Needs more extension.")
    else:
        feedback.append("Arm position: Excellent balance!")

    return " | ".join(feedback)

# Load the video
video = cv2.VideoCapture(r"C:\Users\ayant\Downloads\powell_sprinter_start.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define the required key points
        head = [
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        ]
        butt = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        foot_top = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        # Calculate angles
        body_angle = calculate_angle(head, butt, heel)  # Head to hip to heel angle relative to ground (45° target)
        thigh_angle = calculate_angle(butt, knee, ankle)  # Thigh relative to torso (90° target)
        knee_foot_angle = calculate_angle(knee, ankle, foot_top)  # Knee to foot angle (90° target)
        arm_angle = calculate_angle(shoulder, elbow, wrist)  # Arm extension angle (should be fully extended or nearly so)

        # Provide feedback
        feedback = provide_feedback(body_angle, thigh_angle, knee_foot_angle, arm_angle)

        # Annotate angles on the image
        cv2.putText(frame, f'Body Angle: {int(body_angle)}°',
                    tuple(np.multiply(butt, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Thigh Angle: {int(thigh_angle)}°',
                    tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Knee-Foot Angle: {int(knee_foot_angle)}°',
                    tuple(np.multiply(ankle, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Arm Angle: {int(arm_angle)}°',
                    tuple(np.multiply(shoulder, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Provide overall feedback
        cv2.putText(frame, feedback,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the pose landmarks and connections on the image.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Sprinter Analysis', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video.release()
cv2.destroyAllWindows()
