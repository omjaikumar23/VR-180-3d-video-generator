import os
import shutil
import torch
from PIL import Image
import streamlit as st

from depth_estimator import DepthEstimator
from stereo_renderer import create_stereo_frame
from utils import extract_frames, frames_to_side_by_side_video

# Folders (make sure they exist)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, 'frames', 'left_frames', 'right_frames']:
    os.makedirs(folder, exist_ok=True)

# Load depth model once
depth_model = DepthEstimator(device='cuda' if torch.cuda.is_available() else 'cpu')

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_processing_folders():
    for folder in ['frames', 'left_frames', 'right_frames', PROCESSED_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


# Streamlit UI
st.set_page_config(page_title="2D to VR180 Converter", layout="wide")

banner_img = os.path.join('static', '2D_to_VR180.jpg')
st.image(banner_img, use_column_width=True)

st.title("2D to VR180 Converter: Create Immersive Stereo Videos")

st.markdown("""
Upload any 2D video file (`mp4`, `mov`, `avi`) and let AI convert it into an immersive VR180 stereoscopic video.
""")

uploaded_file = st.file_uploader("Upload 2D Video Clip", type=list(ALLOWED_EXTENSIONS))

if uploaded_file is not None:
    filename = uploaded_file.name
    if not allowed_file(filename):
        st.error("Unsupported file type!")
    else:
        # Save uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded file saved: {filename}")

        if st.button("Convert to VR180"):
            with st.spinner("Processing video, please wait..."):
                clear_processing_folders()

                # Extract frames
                frame_paths = extract_frames(upload_path, 'frames')
                st.write(f"Extracted {len(frame_paths)} frames.")

                left_paths, right_paths = [], []
                progress_bar = st.progress(0)

                for i, frame_path in enumerate(frame_paths):
                    pil_img = Image.open(frame_path).convert('RGB')
                    depth_map = depth_model.estimate_depth(pil_img)
                    left_img, right_img = create_stereo_frame(pil_img, depth_map, eye_separation=15)

                    left_frame_path = f'left_frames/left_{i:05d}.png'
                    right_frame_path = f'right_frames/right_{i:05d}.png'
                    left_img.save(left_frame_path)
                    right_img.save(right_frame_path)

                    left_paths.append(left_frame_path)
                    right_paths.append(right_frame_path)
                    progress_bar.progress((i + 1) / len(frame_paths))

                output_video = os.path.join(PROCESSED_FOLDER, f'vr180_{filename}')
                frames_to_side_by_side_video(left_paths, right_paths, output_video)
                st.success(f"VR180 video created: vr180_{filename}")

                # Display video preview
                video_file = open(output_video, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)

                # Allow download
                st.download_button(
                    label="⬇️ Download Converted VR180 Video",
                    data=video_bytes,
                    file_name=f'vr180_{filename}',
                    mime='video/mp4'
                )
