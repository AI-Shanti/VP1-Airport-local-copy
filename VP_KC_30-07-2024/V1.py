import json
import random
from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify
import cv2
import face_recognition
import numpy as np
import os
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import bson
import base64
import logging

app = Flask(__name__)
app.secret_key = os.urandom(24)
bcrypt = Bcrypt(app)
camera = None

# Database setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition']
users_collection = db['users']

# Define the path to the directory containing the images
image_directory = "static/uploads"

# Define a simple model architecture for emotion detection
class SimpleEmotionModel(nn.Module):
    def __init__(self):
        super(SimpleEmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjust based on the input size
        self.fc2 = nn.Linear(128, 5)  # 5 classes for emotions

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load existing face encodings
def load_face_encodings():
    known_face_encodings = []
    known_face_names = []
    users = users_collection.find({})
    for user in users:
        if 'face_encoding' in user:
            known_face_encodings.append(np.array(user['face_encoding']))
            known_face_names.append(user['name'])
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_face_encodings()

# Load YOLOv5 model
def load_yolov5_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt').to(device)
    with torch.no_grad():
        return model

# Load emotion model
def load_emotion_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEmotionModel()
    try:
        model.load_state_dict(torch.load('KC_weight.pt', map_location=device))
    except FileNotFoundError:
        print("Error: Model file 'emotions.pt' not found.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    model.to(device)
    model.eval()
    return model

model = load_yolov5_model()
emotion_model = load_emotion_model()

# Function to detect faces using YOLOv5
def detect_faces_yolov5(frame, model):
    results = model(frame)
    face_locations = []
    cropped_faces = []
    for *xyxy, conf, cls in results.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        face_locations.append((y1, x2, y2, x1))
        cropped_faces.append(frame[y1:y2, x1:x2])
    return face_locations, cropped_faces

# Function to detect emotions
def detect_emotions(face_image, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = torch.tensor(face_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emotion_preds = model(face_image)
    softmax_scores = nn.functional.softmax(emotion_preds, dim=1)
    emotion_probs = softmax_scores.squeeze().cpu().numpy()
    emotion_label = np.argmax(emotion_probs)
    emotion_map = {1: 'Angry', 0: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprised'}
    detected_emotion = emotion_map.get(emotion_label, "Unknown")
    return detected_emotion

# Encode image to binary
def encode_image_to_binary(image):
    return bson.binary.Binary(cv2.imencode('.jpg', image)[1].tobytes())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/feeds')
def feeds():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return render_template('feeds.html')

@app.route('/stop_feed')
def stop_feed():
    global camera
    if camera:
        camera.release()
    return '', 204

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mobile_feed')
def mobile_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(gen_mobile_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        image = request.files['image']
        
        if name and username and password and image:
            filename = f"{username}.jpg"
            image_path = os.path.join(image_directory, filename)
            image.save(image_path)
            
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]

            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            now = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            users_collection.insert_one({
                'name': name,
                'username': username,
                'password': hashed_password,
                'face_encoding': face_encoding.tolist(),
                'first_seen': now,
                'last_seen': now,
                'visit_count': 0,
                'visits': []
            })

            global known_face_encodings, known_face_names
            known_face_encodings, known_face_names = load_face_encodings()
            
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
    return redirect(url_for('index'))

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    users = users_collection.find({})
    return render_template('profile.html', users=users)

active_sessions = {}

def gen_frames():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names, process_this_frame
    
    process_this_frame = True
    
    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if process_this_frame:
                face_locations, cropped_faces = detect_faces_yolov5(rgb_frame, model)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_names = []
                face_emotions = []
                for face_encoding, (top, right, bottom, left), cropped_face in zip(face_encodings, face_locations, cropped_faces):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        user = users_collection.find_one({'name': name})
                        if user:
                            name_str = str(name).replace("/", "").replace("\\", "").replace(":", "").replace(" ", "")
                            image_path = os.path.join(image_directory, f"{name_str}.jpg")

                            if cropped_face.size != 0:
                                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                                success = cv2.imwrite(image_path, cropped_face)
                                binary_image = encode_image_to_binary(cropped_face)
                                if binary_image is not None:
                                    current_time = datetime.now()
                                    last_image_update = user.get('last_image_update', None)
                                    if last_image_update is None or (current_time - last_image_update).total_seconds() > 10:
                                        users_collection.update_one(
                                            {'name': name},
                                            {'$set': {
                                                'last_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                'face_encoding': face_encoding.tolist(),
                                                'image_data': binary_image,
                                                'last_image_update': current_time
                                            }}
                                        )
                                        if name in active_sessions:
                                            active_sessions[name]['end_time'] = current_time
                                        else:
                                            active_sessions[name] = {
                                                'start_time': current_time,
                                                'end_time': current_time
                                            }
                    else:
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        unidentified_name = f"unidentified_{now}"
                        image_path = os.path.join(image_directory, f"{unidentified_name}.jpg")

                        if cropped_face.size != 0:
                            current_time = datetime.now()
                            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                            success = cv2.imwrite(image_path, cropped_face)
                            binary_image = encode_image_to_binary(cropped_face)
                            if binary_image is not None:
                                users_collection.insert_one({
                                    'name': unidentified_name,
                                    'face_encoding': face_encoding.tolist(),
                                    'first_seen': now,
                                    'last_seen': now,
                                    'image_data': binary_image,
                                    'last_image_update': current_time
                                })
                                known_face_encodings, known_face_names = load_face_encodings()

                    face_emotion = detect_emotions(frame[top:bottom, left:right], emotion_model)
                    face_names.append(name)
                    face_emotions.append(face_emotion)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, face_emotions):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 70), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 36), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_mobile_frames():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names, process_this_frame

    process_this_frame = True

    while True:
        success, frame = mobile_camera.read()  # type: ignore
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if process_this_frame:
                face_locations, cropped_faces = detect_faces_yolov5(rgb_frame, model)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_names = []
                face_emotions = []
                for face_encoding, (top, right, bottom, left), cropped_face in zip(face_encodings, face_locations, cropped_faces):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        user = users_collection.find_one({'name': name})
                        if user:
                            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            users_collection.update_one(
                                {'name': name},
                                {'$set': {'last_seen': now}, '$inc': {'visit_count': 1}}
                            )

                            if name in active_sessions:
                                active_sessions[name]['end_time'] = now
                            else:
                                active_sessions[name] = {
                                    'start_time': now,
                                    'end_time': now
                                }
                    else:
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        unidentified_name = f"unidentified_{now}"
                        image_path = os.path.join(image_directory, f"{unidentified_name}.jpg")

                        face_image = frame[top:bottom, left:right]
                        cv2.imwrite(image_path, face_image)

                        users_collection.insert_one({
                            'name': unidentified_name,
                            'face_encoding': face_encoding.tolist(),
                            'first_seen': now,
                            'last_seen': now,
                            'visit_count': 1,
                            'visits': []
                        })

                        known_face_encodings, known_face_names = load_face_encodings()

                    face_emotion = detect_emotions(frame[top:bottom, left:right], emotion_model)
                    face_names.append(name)
                    face_emotions.append(face_emotion)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, face_emotions):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 70), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 36), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def convert_objectid(data):
    if isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, bson.ObjectId):
        return str(data)
    else:
        return data

# Function to generate random time duration
def generate_random_duration():
    hours = random.randint(0, 5)
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    return str(timedelta(hours=hours, minutes=minutes, seconds=seconds))

# Update each record with a new field 'avg_visit_duration'
records = users_collection.find()
for record in records:
    avg_duration = generate_random_duration()
    users_collection.update_one({'_id': record['_id']}, {'$set': {'avg_visit_duration': avg_duration}})

print("Records updated with avg_visit_duration")

@app.route('/export_users', methods=['GET'])
def export_users():
    return render_template('export_users.html')

@app.route('/perform_export', methods=['POST'])
def perform_export():
    def separate_users_to_json():
        users = list(users_collection.find({}))
        known_users = []
        unknown_users = []

        for user in users:
            if user['name'].startswith('unidentified_') or user['name'] == "Unknown":
                unknown_users.append(user)
            else:
                known_users.append(user)

        with open('known_users.json', 'w') as known_file:
            json.dump(known_users, known_file, default=str)

        with open('unknown_users.json', 'w') as unknown_file:
            json.dump(unknown_users, unknown_file, default=str)

    separate_users_to_json()
    return jsonify({'message': 'Users exported successfully.'})

@app.route('/known_visitors', methods=['GET'])
def known_visitors():
    try:
        with open('known_users.json', 'r') as file:
            known_users = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        known_users = []
    return render_template('known_users.html', users=known_users)

@app.route('/unknown_visitors', methods=['GET'])
def unknown_visitors():
    try:
        with open('unknown_users.json', 'r') as file:
            unknown_users = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        unknown_users = []
    return render_template('unknown_users.html', users=unknown_users)

@app.route('/known_visitors_secure')
def known_visitors_secure():
    return render_template('known_visitors_secure.html')

@app.route('/unknown_visitors_secure')
def unknown_visitors_secure():
    return render_template('unknown_visitors_secure.html')

@app.route('/maps')
def maps():
    return render_template('maps.html')

def parse_datetime(dt_str):
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H:%M"]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/filter', methods=['GET'])
def filter_records():
    start_date = request.args.get('start_date')
    start_time = request.args.get('start_time')
    end_date = request.args.get('end_date')
    end_time = request.args.get('end_time')

    if not start_date or not start_time or not end_date or not end_time:
        return "Please provide start and end date and time", 400

    start_datetime = parse_datetime(f"{start_date} {start_time}")
    end_datetime = parse_datetime(f"{end_date} {end_time}")

    if not start_datetime or not end_datetime:
        return "Invalid datetime format provided", 400

    records = users_collection.find()
    filtered_records = []
    for record in records:
        first_seen = record.get('first_seen', '')
        record_first_seen = parse_datetime(first_seen)
        if record_first_seen and start_datetime <= record_first_seen <= end_datetime:
            filtered_records.append(record)

    total_visit_count = sum(record.get('visit_count', 0) for record in filtered_records)
    return jsonify(total_visit_count)

@app.route('/count_known_unknown', methods=['GET'])
def count_known_unknown():
    start_date = request.args.get('start_date')
    start_time = request.args.get('start_time')
    end_date = request.args.get('end_date')
    end_time = request.args.get('end_time')

    if not start_date or not start_time or not end_date or not end_time:
        return jsonify({'error': 'Please provide start and end date and time'}), 400

    start_datetime = parse_datetime(f"{start_date} {start_time}")
    end_datetime = parse_datetime(f"{end_date} {end_time}")

    if not start_datetime or not end_datetime:
        return jsonify({'error': 'Invalid datetime format provided'}), 400

    records = users_collection.find()
    known_persons_count = 0
    unknown_persons_count = 0

    for record in records:
        first_seen = record.get('first_seen', '')
        record_first_seen = parse_datetime(first_seen)
        name = record.get('name', '')

        if record_first_seen and start_datetime <= record_first_seen <= end_datetime:
            if record.get('is_known', False):
                known_persons_count += 1
            else:
                if name.lower().startswith('unidentified'):
                    unknown_persons_count += 1
                else:
                    known_persons_count += 1

    return jsonify({
        'known_persons_count': known_persons_count,
        'unknown_persons_count': unknown_persons_count
    })

@app.route('/details', methods=['GET'])
def get_details():
    start_date = request.args.get('start_date')
    start_time = request.args.get('start_time')
    end_date = request.args.get('end_date')
    end_time = request.args.get('end_time')
    person_type = request.args.get('type')

    if not start_date or not start_time or not end_date or not end_time or not person_type:
        return jsonify({'error': 'Missing required parameters'}), 400

    start_datetime = parse_datetime(f"{start_date} {start_time}")
    end_datetime = parse_datetime(f"{end_date} {end_time}")

    if not start_datetime or not end_datetime:
        return jsonify({'error': 'Invalid datetime format provided'}), 400

    records = users_collection.find()
    details = []
    for record in records:
        first_seen = record.get('first_seen', '')
        record_first_seen = parse_datetime(first_seen)
        name = record.get('name', '')

        if record_first_seen and start_datetime <= record_first_seen <= end_datetime:
            if person_type == 'known' and not name.lower().startswith('unidentified'):
                details.append(record)
            elif person_type == 'unknown' and name.lower().startswith('unidentified'):
                details.append(record)

    details = convert_objectid(details)
    for detail in details:
        if 'image_data' in detail:
            detail['image_data'] = base64.b64encode(detail['image_data']).decode('utf-8')

    return jsonify(details)

@app.route('/avg_visit_duration')
def avg_visit_duration():
    try:
        start_date_str = request.args.get('start_date')
        start_time_str = request.args.get('start_time')
        end_date_str = request.args.get('end_date')
        end_time_str = request.args.get('end_time')

        if not (start_date_str and start_time_str and end_date_str and end_time_str):
            return jsonify({'error': 'Missing required parameters'}), 400

        start_dt = datetime.strptime(f"{start_date_str} {start_time_str}", "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(f"{end_date_str} {end_time_str}", "%Y-%m-%d %H:%M:%S")

        records = users_collection.find({
            '$or': [
                {'first_seen': {'$gte': start_dt, '$lte': end_dt}},
                {'last_seen': {'$gte': start_dt, '$lte': end_dt}}
            ]
        })

        total_duration = timedelta()
        count = 0

        for record in records:
            if 'first_seen' in record and 'last_seen' in record:
                first_seen = datetime.strptime(record['first_seen'], "%Y-%m-%d %H:%M")
                last_seen = datetime.strptime(record['last_seen'], "%Y-%m-%d %H:%M:%S")
                visit_duration = last_seen - first_seen
                total_duration += visit_duration
                count += 1

        if count > 0:
            average_duration = total_duration / count
            return jsonify({'average_visit_duration': str(average_duration)})
        else:
            return jsonify({'average_visit_duration': 'No visits found'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
