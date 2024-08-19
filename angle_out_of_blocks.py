import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def calculate_vertical_distance_from_bottom(frame_height, y_coordinate):
    """Calculate the vertical distance from the bottom of the frame to a point's y-coordinate."""
    return frame_height - y_coordinate


def calculate_angle(point1, point2):
    """Calculate the angle between the line formed by two points and the horizontal axis (ground)."""
    delta_y = point1[1] - point2[1]
    delta_x = point2[0] - point1[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return abs(angle)


def annotate_landmark(frame, landmark, frame_width, frame_height, label):
    """Annotate a landmark on the frame with its coordinates."""
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    cv2.putText(frame, f'{label}: ({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return (x, y)


def draw_horizontal_line(frame, y_coordinate):
    """Draw a horizontal line at the given y-coordinate."""
    frame_height, frame_width, _ = frame.shape
    cv2.line(frame, (0, y_coordinate), (frame_width, y_coordinate), (255, 0, 0), 2)


# Load the video
video = cv2.VideoCapture(r"C:\Users\ayant\Downloads\powell_sprinter_start (online-video-cutter.com).mp4")

previous_distance = None
arm_dropping_frame = None
start_time = time.time()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Process the image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Check if all relevant landmarks are detected
        visibility_threshold = 0.5
        if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > visibility_threshold and
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > visibility_threshold):

            # All required features are detected
            wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height

            # Calculate the vertical distance from the bottom of the frame to the wrist
            current_distance = calculate_vertical_distance_from_bottom(frame_height, wrist_y)

            # Annotate key points for both legs and shoulders
            left_ankle = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], frame_width, frame_height,
                                           "Left Ankle")
            left_knee = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.LEFT_KNEE], frame_width, frame_height,
                                          "Left Knee")
            left_hip = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.LEFT_HIP], frame_width, frame_height,
                                         "Left Hip")
            left_shoulder = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], frame_width,
                                              frame_height, "Left Shoulder")
            right_ankle = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE], frame_width,
                                            frame_height, "Right Ankle")
            right_knee = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], frame_width, frame_height,
                                           "Right Knee")
            right_hip = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.RIGHT_HIP], frame_width, frame_height,
                                          "Right Hip")
            right_shoulder = annotate_landmark(frame, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], frame_width,
                                               frame_height, "Right Shoulder")

            # Determine which ankle is behind (non-lead leg)
            if left_ankle[1] > right_ankle[1]:  # The ankle with a higher y-coordinate is behind
                behind_ankle = left_ankle
                corresponding_shoulder = left_shoulder
                side = "Left"
            else:
                behind_ankle = right_ankle
                corresponding_shoulder = right_shoulder
                side = "Right"

            # Draw horizontal line at the behind ankle
            draw_horizontal_line(frame, behind_ankle[1])

            # Draw a line from the behind ankle to the corresponding shoulder
            cv2.line(frame, behind_ankle, corresponding_shoulder, (0, 255, 255), 2)

            # Calculate and annotate the angle between the horizontal line and the ankle-shoulder line
            horizontal_angle = calculate_angle(behind_ankle, corresponding_shoulder)
            cv2.putText(frame, f'{side} Ankle-Shoulder Horizontal Angle: {horizontal_angle:.2f}°',
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Log the distances for debugging
            print(
                f"Current Vertical Distance: {current_distance:.4f}, Previous Vertical Distance: {previous_distance:.4f}" if previous_distance else f"Current Vertical Distance: {current_distance:.4f}")

            # Check time interval (every 0.2 seconds)
            current_time = time.time()
            if current_time - start_time >= 0.2:
                start_time = current_time

                if previous_distance is not None:
                    # Check if the vertical distance is decreasing (arm is dropping)
                    if current_distance < previous_distance:
                        arm_dropping_frame = frame.copy()
                        cv2.imwrite(r'C:\Users\ayant\Downloads\arm_drop_start.png', arm_dropping_frame)
                        print(f"Screenshot saved successfully at C:\\Users\\ayant\\Downloads\\arm_drop_start.png with {side} Ankle-Shoulder Horizontal Angle: {horizontal_angle:.2f}°.")
                        break  # Exit loop after capturing the image

                previous_distance = current_distance

            # Annotate vertical distance on the image
            cv2.putText(frame, f'Wrist Vertical Distance: {current_distance:.2f}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw the pose landmarks and connections on the image.
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            print("Waiting for all features to be detected...")

    # Display the frame
    cv2.imshow('Sprinter Analysis', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Ensure features were detected before saving
if arm_dropping_frame is not None:
    cv2.imwrite(r'C:\Users\ayant\Downloads\arm_drop_start.png', arm_dropping_frame)
else:
    print("No arm drop detected or insufficient feature detection.")

# Release the video capture object and close the window
video.release()
cv2.destroyAllWindows()
