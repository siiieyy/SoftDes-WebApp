from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Set a hardcoded secret key for session management
app.secret_key = "8e3eba505815cee3e0ddd65cba77525c12016df8ee8cc4c03afd32aa285b1675"

# Load both YOLO models
model_1 = YOLO("best3.pt")
model_2 = YOLO("bestModel.pt")

# Open the camera
camera = cv2.VideoCapture(0)

# Variable to store detected labels
detected_labels = []

# Hardcoded user credentials
USER_CREDENTIALS = {
    "admin": "admin"
}

def generate_frames():
    while True:
        success, frame = camera.read()  # Read a frame from the camera
        if not success:
            break

        # Convert the frame to RGB for YOLO models
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run both YOLO models
        results_1 = model_1(frame_rgb)
        results_2 = model_2(frame_rgb)

        # Extract detected labels and their counts for model 1
        labels_1 = [results_1[0].names[int(label)] for label in results_1[0].boxes.cls]
        
        # Extract detected labels and their counts for model 2
        labels_2 = [results_2[0].names[int(label)] for label in results_2[0].boxes.cls]

        # Combine the labels from both models
        combined_labels = labels_1 + labels_2
        label_counts = {label: combined_labels.count(label) for label in set(combined_labels)}

        # Prepare the text to display below the webcam frame (as a log)
        global detected_labels
        detected_labels = [f"{count} {label}" for label, count in label_counts.items()]

        # Generate annotated frames for both models and convert them back to BGR
        annotated_frame_1 = cv2.cvtColor(results_1[0].plot(), cv2.COLOR_RGB2BGR)
        annotated_frame_2 = cv2.cvtColor(results_2[0].plot(), cv2.COLOR_RGB2BGR)

        # Combine both frames with bounding boxes
        combined_frame = cv2.addWeighted(annotated_frame_1, 0.5, annotated_frame_2, 0.5, 0)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()

        # Yield the frame as a video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def login():
    # Render the login page
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    # Validate login credentials
    username = request.form.get('username')
    password = request.form.get('password')
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        session['logged_in'] = True
        return redirect(url_for('index'))
    return render_template('login.html', error="Invalid username or password")

@app.route('/logout')
def logout():
    # Clear the session and redirect to login page
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def index():
    # Check if the user is logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    # Load the HTML file
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Stream the processed video to the webpage
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    # Return the detected labels as a JSON response
    return jsonify(detected_labels)

if __name__ == "__main__":
    app.run(debug=True)
