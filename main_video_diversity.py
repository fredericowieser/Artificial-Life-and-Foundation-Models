import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Corrected import statement (remove the .py extension)
from AskVideos_VideoCLIP import video_clip

def load_video_frames(filepath, max_frames=20):
    """
    Loads frames from a video file using OpenCV.
    Returns a list of PIL Images.
    """
    frames = []
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Choose frames at regular intervals if the video is longer than max_frames.
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = range(total_frames)
    
    idx_set = set(frame_indices)
    current_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in idx_set:
            # Convert BGR to RGB and then to a PIL Image.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)
        current_idx += 1
        if current_idx > max(frame_indices):
            break
            
    cap.release()
    return frames

def main():
    # 1. Set your video directory and collect video file paths.
    video_dir = "/graph_flavour/test"  # <-- change this to your directory
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"Found {len(video_files)} video(s).")
    
    if len(video_files) < 2:
        print("Need at least 2 videos to compute similarities. Exiting.")
        return

    # 2. Load the model and visual processor using the evaluation configuration.
    eval_config = '/workspace/Artificial-Life-and-Foundation-Models/AskVideos_VideoCLIP/eval_configs/video_clip_v0.1.yaml'
    model, vis_processor = video_clip.load_model(eval_config)

    # 3. For each video file, load frames (as a list of PIL Images).
    videos = []  # list of videos, each video is a list of frames.
    for vid_path in video_files:
        frames = load_video_frames(vid_path, max_frames=20)
        videos.append(frames)

    # 4. Compute video embeddings.
    # get_all_video_embeddings returns a tensor of shape 
    # [num_videos, clip_dim_size, query_tokens] (e.g., [N, 1024, 32]).
    video_embs = video_clip.get_all_video_embeddings(videos, model, vis_processor)
    
    # For computing pairwise dot product similarities,
    # average the embeddings over the clip_dim and query_tokens to get one vector per video.
    video_vectors = video_embs.mean(dim=[1, 2])  # shape: [num_videos, D]
    similarities = video_vectors @ video_vectors.T  # dot product matrix [N, N]
    
    print("Video embeddings shape:", video_embs.shape)
    print("Video vectors shape:", video_vectors.shape)
    print("Similarity matrix shape:", similarities.shape)

    # 5. Prepare data for scatter plot.
    N = similarities.shape[0]
    x_vals, y_vals, sim_vals = [], [], []
    for i in range(N):
        for j in range(N):
            x_vals.append(i)
            y_vals.append(j)
            sim_vals.append(similarities[i, j].item())
    
    # 6. Plot the scatter plot.
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(x_vals, y_vals, c=sim_vals, cmap='viridis')
    plt.colorbar(scatter, label="Dot Product Similarity")
    plt.title("Pairwise Video Similarities")
    plt.xlabel("Video Index")
    plt.ylabel("Video Index")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
