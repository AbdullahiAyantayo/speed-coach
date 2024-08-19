import base64
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Ensure the screenshots directory exists
screenshots_dir = "static/screenshots"
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)


@app.route('/')
def home():
    print("Home route accessed")
    return render_template('home.html')


@app.route('/angle_out_of_blocks')
def angle_out_of_blocks():
    print("Angle out of blocks accessed")
    return render_template('angle_out_of_blocks.html')


@app.route('/pre_acceleration', methods=['GET', 'POST'])
def pre_acceleration():
    print("Pre acceleration accessed")
    if request.method == 'POST':
        video_data = request.form.get('video_data')

        if video_data:
            # Decode the base64 video data
            video_data = video_data.split(',')[1]  # Remove the data type prefix
            video_bytes = base64.b64decode(video_data)
            video_path = os.path.join(screenshots_dir, 'recorded_video.webm')

            # Save the video to a file
            with open(video_path, 'wb') as video_file:
                video_file.write(video_bytes)

            # Call the pre_acceleration function here and pass the video path
            # Example: result = analyze_pre_acceleration(video_path)

            # Then render the result or save a screenshot, etc.
            return render_template('pre_acceleration.html', result="Analysis Complete")  # Replace with actual result

    return render_template('pre_acceleration.html')


if __name__ == "__main__":
    app.run(debug=True)



'''
from flask import Flask, render_template, Response, jsonify
import cv2
import time
import numpy as np
import pyttsx3
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reaction_times.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Database model for storing reaction times
class ReactionTime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.Float, nullable=False)


# Initialize the database
db.create_all()

# Global variables
cap = None
start_time = None


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    global cap, start_time
    cap = cv2.VideoCapture(0)

    # Countdown
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Get Ready", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        time.sleep(1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Show "Go!" and voice it out
    speak("Go!")
    start_time = time.time()
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Go!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Movement detection
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        non_zero_count = np.count_nonzero(diff)

        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Time: {elapsed_time:.3f} s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        if non_zero_count > 1000:
            reaction_time = elapsed_time
            new_time = ReactionTime(time=reaction_time)
            db.session.add(new_time)
            db.session.commit()
            break

        prev_gray = gray

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/results')
def results():
    times = ReactionTime.query.all()
    return jsonify([{'id': t.id, 'time': t.time} for t in times])


if __name__ == '__main__':
    app.run(debug=True)

'''