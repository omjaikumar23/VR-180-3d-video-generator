import cv2
from PIL import Image
import os
import numpy as np
import subprocess

def extract_frames(input_video_path, frame_dir='frames'):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    cap = cv2.VideoCapture(input_video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f'frame_{idx:05d}.png')
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        idx += 1
    cap.release()
    return frames

def frames_to_side_by_side_video(left_frames, right_frames, output_path, fps=30):
    assert len(left_frames) == len(right_frames)
    first_frame = Image.open(left_frames[0])
    width, height = first_frame.size
    out_width = width * 2
    out_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))

    for l_frame_path, r_frame_path in zip(left_frames, right_frames):
        l_img = cv2.imread(l_frame_path)
        r_img = cv2.imread(r_frame_path)
        combined = np.hstack((l_img, r_img))
        out.write(combined)
    out.release()

    # Convert to H.264 for browser compatibility
    convert_to_h264(temp_output, output_path)
    os.remove(temp_output)

def convert_to_h264(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    print("Running FFmpeg to convert to H.264...")
    subprocess.run(cmd, check=True)



