import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Import your VideoCLIP class
from asal_pytorch.foundation_models.video_clip import VideoCLIP  # <-- Replace with the actual filename if needed

def load_video_frames(filepath, max_frames=20):
    """
    Loads frames from a video file using OpenCV, subsampling up to 'max_frames'.
    Returns a list of frames as PIL Images or NumPy arrays (compatible with VideoCLIP).
    """
    frames = []
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If max_frames is less than total, pick frames at regular intervals
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = range(total_frames)
    
    idx_set = set(frame_indices)  # for quick membership test
    current_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_idx in idx_set:
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        current_idx += 1
        if current_idx > frame_indices[-1]:
            break
    
    cap.release()
    return frames

def main():
    # 2. Specify your video directory
    video_dir = "/Users/baidn/Artificial-Life-and-Foundation-Models/graph_flavour/test"  # <--- change to your directory
    
    # 3. Collect all video file paths
    # Adjust the glob pattern if your videos are in different formats
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"Found {len(video_files)} video(s).")
    
    if len(video_files) < 2:
        print("Need at least 2 videos to compute pairwise similarities. Exiting.")
        return

    # 4. Initialize the VideoCLIP model
    model = VideoCLIP(
        model_name="m-bain/videoclip",
        device=None,       # automatically chooses GPU if available
        torch_dtype=torch.float16
    )
    
    # 5. Compute embeddings for each video
    embeddings = []
    for vid_path in video_files:
        frames = load_video_frames(vid_path, max_frames=20)
        emb = model.embed_video(frames)  # shape [1, D]
        embeddings.append(emb)
    
    # Stack all embeddings into one tensor of shape [N, D]
    embeddings = torch.cat(embeddings, dim=0)  # N x D
    
    # 6. Compute all pairwise dot products (similarities)
    # Because embeddings are L2-normalized, dot product == cosine similarity
    similarities = embeddings @ embeddings.T  # shape [N, N]
    
    # 7. Prepare data for scatter plot
    # We'll make a list of (i, j, sim_ij) for all i,j in [0..N-1].
    # i -> x-axis, j -> y-axis, similarity -> color
    N = similarities.shape[0]
    x_vals = []
    y_vals = []
    sim_vals = []
    
    for i in range(N):
        for j in range(N):
            x_vals.append(i)
            y_vals.append(j)
            sim_vals.append(similarities[i, j].item())
    
    # 8. Plot scatter of (i, j) with color = similarity
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(x_vals, y_vals, c=sim_vals, cmap='viridis')
    plt.colorbar(scatter, label="Similarity")
    plt.title("Pairwise Video Similarities (Dot Products)")
    plt.xlabel("Video Index i")
    plt.ylabel("Video Index j")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
