import os
import json
from collections import deque

import imageio
import jax.numpy as jnp

from asal.foundation_models import CLIP
import pandas as pd
def calc_average_pairwise_oe_score(z):
    """
    Calculates the average pairwise open-endedness score for a set of embeddings.
    For each unique pair in z (shape (N, D)), it computes the dot product, and then
    averages these values. A lower score indicates more diversity among embeddings.
    """
    kernel = z @ z.T  # Compute pairwise dot products, shape (N, N)
    # Create a mask for the lower triangle (unique pairs only)
    mask = jnp.tril(jnp.ones_like(kernel), k=-1)
    sum_pairs = (kernel * mask).sum()
    num_pairs = mask.sum()
    return sum_pairs / num_pairs

def gather_videos_by_level(root_node):
    """
    Given the root node of your JSON tree, returns a dictionary mapping:
      level -> list of video paths
    """
    queue = deque([(root_node, 0)])
    videos_by_level = {}
    while queue:
        node, level = queue.popleft()
        if "video_path" in node:
            videos_by_level.setdefault(level, []).append(node["video_path"])
        for child in node.get("children", []):
            queue.append((child, level + 1))
    return videos_by_level

def compute_oe_score_for_videos(video_paths, foundation_model, fraction=1.0):
    """
    Given a list of MP4 video paths, extracts the frame at the specified fraction
    of the video (e.g., fraction=1.0 for final frame, 0.25 for 25% in) from each video,
    computes embeddings with the given model, then calculates the average pairwise
    open-endedness score across all videos.
    
    Parameters:
      video_paths: list of paths to MP4 files.
      foundation_model: an object (e.g., CLIP) with a method embed_img()
      fraction: float in (0, 1] indicating which frame to sample.
      
    Returns:
      float or None: the computed score, or None if no frames could be processed.
    """
    embeddings = []
    for vpath in video_paths:
        if not os.path.isfile(vpath):
            print(f"Warning: video file not found: {vpath}")
            continue
        try:
            reader = imageio.get_reader(vpath, format="ffmpeg")
        except Exception as e:
            print(f"Could not open {vpath}: {e}")
            continue

        frames = []
        try:
            for frame in reader:
                frames.append(frame)
        except Exception as e:
            print(f"Error reading frames from {vpath}: {e}")
            continue

        if not frames:
            continue

        # Calculate the frame index based on the given fraction.
        index = int(fraction * (len(frames) - 1))
        selected_frame = frames[index]

        # Normalize the image to [0,1] and convert to a JAX array.
        img_jax = jnp.array(selected_frame) / 255.0
        embedding = foundation_model.embed_img(img_jax)
        embeddings.append(embedding)

    if not embeddings:
        return None

    zs = jnp.stack(embeddings, axis=0)
    score = calc_average_pairwise_oe_score(zs)
    return float(score)

def main():
    # Load the tree structure from JSON.
    tree_json_path = "branches.json"  # update this to your tree JSON path
    with open(tree_json_path, "r", encoding="utf-8") as f:
        root_node = json.load(f)

    # Gather video paths grouped by tree level.
    videos_by_level = gather_videos_by_level(root_node)

    # Initialize the CLIP model.
    fm = CLIP()

    # Define the fractions to sample (final frame, 25%, 50%, 75%).
    fractions = {
        "final": 1.0,
        "0.25": 0.25,
        "0.5": 0.5,
        "0.75": 0.75,
    }

    # Compute the average pairwise OE score per level for each fraction.
    level_scores = {}
    for level in sorted(videos_by_level.keys()):
        video_paths = videos_by_level[level]
        scores_for_level = {}
        for label, frac in fractions.items():
            score = compute_oe_score_for_videos(video_paths, fm, fraction=frac)
            scores_for_level[label] = score if score is not None else 0.0
            print(f"Level {level} - {label} frame Average Pairwise OE Score: {scores_for_level[label]}")
        level_scores[level] = scores_for_level

    # Save the scores to a JSON file.
    out_json = "level_oe_scores.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(level_scores, f, indent=4)
    print(f"Saved scores to {out_json}")

    df = pd.DataFrame.from_dict(level_scores, orient="index")
    df.index.name = "Level"
    print("\nAverage Pairwise OE Score Table:")
    print(df)

    # Save the table as a CSV file.
    csv_path = "level_oe_scores_table.csv"
    df.to_csv(csv_path)
    print(f"Saved table to {csv_path}")

if __name__ == "__main__":
    main()
