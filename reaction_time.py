import cv2
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import numpy as np

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = r"C:\Users\ayant\Downloads\noah_lyles_whistle_isolated.wav"
    audio.write_audiofile(audio_path)
    return audio_path

def detect_start_signal(audio_path, trigger_words=["go", "start"]):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio).lower()
        print(f"Transcript: {transcript}")  # Debugging: print the recognized text
        for word in trigger_words:
            if word in transcript:
                return True
        return False
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return False

def detect_movement(video_path, start_time, sensitivity=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the video file")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        non_zero_count = np.count_nonzero(diff)

        if non_zero_count > sensitivity:
            movement_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.release()
            return movement_time - start_time

        prev_gray = gray

    cap.release()
    return None

def calculate_reaction_time(video_path):
    # Step 1: Extract audio from video
    audio_path = extract_audio_from_video(video_path)

    # Step 2: Detect the start signal
    start_signal_detected = detect_start_signal(audio_path)
    if not start_signal_detected:
        print("Start signal not detected.")
        return None

    # Step 3: Detect movement after the start signal
    cap = cv2.VideoCapture(video_path)
    ret, _ = cap.read()
    start_time = 0
    if ret:
        start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()

    reaction_time = detect_movement(video_path, start_time)
    if reaction_time is not None:
        print(f"Reaction time: {reaction_time:.3f} seconds")
    else:
        print("Could not detect movement.")
        return None

    # Display video with reaction time overlay
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay reaction time on the video
        if reaction_time is not None:
            cv2.putText(frame, f"Reaction Time: {reaction_time:.3f} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return reaction_time

# Example usage
video_path = r"C:\Users\ayant\Downloads\noah_lyles_start.mp4"
reaction_time = calculate_reaction_time(video_path)
