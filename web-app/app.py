from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load both YOLO models
model_1 = YOLO("best3.pt")
model_2 = YOLO("bestModel.pt")

# Open the camera
camera = cv2.VideoCapture(0)

# Variable to store detected labels
detected_labels = []

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

        # If you want to use different colors for the bounding boxes:
        for box in results_1[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame_1, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for Model 1

        for box in results_2[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame_2, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for Model 2

        # Combine both frames with bounding boxes
        combined_frame = cv2.addWeighted(annotated_frame_1, 0.5, annotated_frame_2, 0.5, 0)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()

        # Yield the frame as a video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Load the HTML file (no model selection required)
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
