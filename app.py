import os
import shutil
import torch
from flask import Flask, render_template, request, send_file, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from depth_estimator import DepthEstimator
from stereo_renderer import create_stereo_frame
from utils import extract_frames, frames_to_side_by_side_video

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

depth_model = DepthEstimator(device='cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for folder in ['frames', 'left_frames', 'right_frames', PROCESSED_FOLDER]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

        if 'video' not in request.files:
            return "No file part"
        file = request.files['video']
        if file.filename == '':
            return "No file selected"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            print(f"Extracting frames from {filename}...")
            frame_paths = extract_frames(filepath, 'frames')
            print(f"Total frames extracted: {len(frame_paths)}")

            left_paths, right_paths = [], []

            for i, frame_path in enumerate(frame_paths):
                print(f"Processing frame {i+1}/{len(frame_paths)}: {frame_path}")
                pil_img = Image.open(frame_path).convert('RGB')
                depth_map = depth_model.estimate_depth(pil_img)
                left_img, right_img = create_stereo_frame(pil_img, depth_map, eye_separation=15)
                left_frame_path = f'left_frames/left_{i:05d}.png'
                right_frame_path = f'right_frames/right_{i:05d}.png'
                left_img.save(left_frame_path)
                right_img.save(right_frame_path)
                left_paths.append(left_frame_path)
                right_paths.append(right_frame_path)
                print(f"Saved stereo frames for frame {i+1}")

            output_video = os.path.join(PROCESSED_FOLDER, f'vr180_{filename}')
            print("Generating VR180 video...")
            frames_to_side_by_side_video(left_paths, right_paths, output_video)
            print(f"Video saved at: {output_video}")

            return redirect(url_for('preview', filename=f'vr180_{filename}'))

    return render_template('index.html')

@app.route('/preview/<filename>')
def preview(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found"
    return render_template('preview.html', video_file=filename)

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

@app.route('/processed/<path:filename>')
def processed_files(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
