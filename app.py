from flask import Flask, render_template, request, redirect, url_for
import requests
import cv2
import os

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Model server URL (from Vercel environment variable)
MODEL_SERVER_URL = 'https://civic-anthropomorphically-viola.ngrok-free.dev'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Send image to model server (your ngrok URL)
        with open(filepath, 'rb') as f:
            files = {'image': ('image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{MODEL_SERVER_URL}/detect", files=files)
            response.raise_for_status()
            data = response.json()
        
        if not data.get('success'):
            return render_template('index.html', error=data.get('error', 'Detection failed'))
        
        # Draw bounding boxes on image
        img = cv2.imread(filepath)
        
        for det in data['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class']
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with background
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save result image
        result_img_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        cv2.imwrite(result_img_path, img)

        return render_template('index.html',
                             uploaded_image=url_for('static', filename='uploads/' + file.filename),
                             result_image=url_for('static', filename='results/' + file.filename),
                             detection_count=data['count'])
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Could not connect to model server: {str(e)}"
        return render_template('index.html', error=error_msg)
    
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return render_template('index.html', error=error_msg)

if __name__ == "__main__":
    app.run(debug=True, port=5003)
