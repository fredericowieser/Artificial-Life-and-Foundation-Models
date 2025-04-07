import os
import json
from collections import deque

import imageio
import jax.numpy as jnp

from asal.foundation_models import CLIP
from asal.asal_metrics import calc_open_endedness_score


def gather_videos_by_level(root_node):
    """
    Given the root node of your JSON tree, returns a dictionary:
      {
        0: [list of video paths at level 0],
        1: [list of video paths at level 1],
        2: [list of video paths at level 2],
        ...
      }
    """
    queue = deque([(root_node, 0)])
    videos_by_level = {}
    while queue:
        node, level = queue.popleft()
        # Record the current node's video_path if available
        if "video_path" in node:
            videos_by_level.setdefault(level, []).append(node["video_path"])
        # Enqueue children with the level incremented by 1
        for child in node.get("children", []):
            queue.append((child, level + 1))
    return videos_by_level


def compute_oe_score_for_level(video_paths, foundation_model):
    """
    Given a list of MP4 video paths, extracts the final frame from each video,
    computes embeddings with the given model, and returns a single OE score
    for the entire set.
    """
    embeddings = []

    for vpath in video_paths:
        if not os.path.isfile(vpath):
            print(f"Warning: video file not found: {vpath}")
            continue

        try:
            # Read the MP4 file using imageio with the ffmpeg backend.
            reader = imageio.get_reader(vpath, format="ffmpeg")
        except Exception as e:
            print(f"Could not open {vpath}: {e}")
            continue

        final_frame = None
        try:
            # Iterate through the video frames to get the final frame.
            for frame in reader:
                final_frame = frame
        except Exception as e:
            print(f"Error reading frames from {vpath}: {e}")
            continue

        if final_frame is None:
            continue

        # Normalize the final frame to [0, 1] and convert it to a JAX array.
        img_jax = jnp.array(final_frame) / 255.0
        embedding = foundation_model.embed_img(img_jax)
        embeddings.append(embedding)

    if not embeddings:
        return None

    # Stack all embeddings and compute the open-endedness (diversity) score.
    zs = jnp.stack(embeddings, axis=0)
    score = calc_open_endedness_score(zs)
    return float(score)


def main():
    # Load your tree structure from JSON.
    # Make sure to update the file path to where your tree JSON is stored.
    tree_json_path = "graphs/a_catepillar_lenia/branches.json"
    with open(tree_json_path, "r", encoding="utf-8") as f:
        root_node = json.load(f)

    # Gather video paths by tree level using BFS.
    videos_by_level = gather_videos_by_level(root_node)

    # Initialize the CLIP model.
    fm = CLIP()

    # Compute a single open-endedness (OE) score per level.
    level_scores = {}
    for level in sorted(videos_by_level.keys()):
        video_paths = videos_by_level[level]
        score = compute_oe_score_for_level(video_paths, fm)
        if score is None:
            score = 0.0
        level_scores[level] = score
        print(f"Level {level} OE Score: {score}")

    # Save the level scores to a JSON file.
    out_json = "level_oe_scores.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(level_scores, f, indent=4)
    print(f"Saved scores to {out_json}")


if __name__ == "__main__":
    main()
