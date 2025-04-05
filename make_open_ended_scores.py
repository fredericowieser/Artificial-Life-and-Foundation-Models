import os
import re
import json
import imageio
import jax.numpy as jnp
import numpy as np

from asal.foundation_models import CLIP
from asal.asal_metrics import calc_open_endedness_score


def parse_config(config_path):
    """
    Reads a key:value config file (e.g. config.txt) and returns a dictionary of config parameters.
    Each line is expected to contain 'key: value'.
    Lines without a colon or empty lines are ignored.

    Parameters
    ----------
    config_path : str
        The path to the config file.

    Returns
    -------
    dict
        A dictionary of configuration parameters. Keys and values are strings.
    """
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    config[key.strip()] = val.strip()
    return config


def compute_gif_oe_scores(directory, foundation_model):
    """
    Computes open-endedness scores for each iteration's GIF by:
      1. Reading prompts.txt to get the iteration order.
      2. Sampling a fixed number of frames from each GIF (based on OE_SAMPLES from config.txt).
      3. Embedding the sampled frames with a given foundation model (CLIP).
      4. Calculating the open-endedness score for each prompt's GIF.
      5. Saving results in a JSON file, e.g. 'oe_scores.json'.

    Parameters
    ----------
    directory : str
        Path to a directory containing config.txt, prompts.txt, and 'media/videos' with iteration_<n>_*.gif files.
    foundation_model : Any
        A foundation model object with an 'embed_img()' method (e.g., an instance of CLIP).

    Returns
    -------
    None
        Results are written to 'oe_scores.json' in 'directory'.
    """
    oe_samples = 32  # Number of frames to sample for open-endedness score

    # Load prompts
    prompts_file = os.path.join(directory, "prompts.txt")
    if not os.path.isfile(prompts_file):
        print(f"Warning: No prompts.txt found in {directory}. Skipping.")
        return
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Find GIF files for each iteration
    media_dir = os.path.join(directory, "media", "videos")
    gif_files = [fname for fname in os.listdir(media_dir) if fname.endswith(".gif")]
    iteration_to_gif = {}
    for fname in gif_files:
        match = re.match(r"^iteration_(\d+)_.*\.gif$", fname)
        if match:
            # iteration_1 maps to index=0, iteration_2 -> index=1, etc.
            idx = int(match.group(1)) - 1
            iteration_to_gif[idx] = os.path.join(media_dir, fname)

    # Compute and store open-endedness scores for each prompt's GIF
    oe_scores_per_prompt = {}
    for i, prompt in enumerate(prompts):
        # If there's no GIF for this iteration index, skip
        if i not in iteration_to_gif:
            continue

        gif_path = iteration_to_gif[i]
        all_frames = imageio.mimread(gif_path)
        if not all_frames:
            continue

        # Sample 'oe_samples' frames evenly from all_frames
        total_frames = len(all_frames)
        sample_indices = np.linspace(0, total_frames - 1, oe_samples, dtype=int)
        sampled_frames = [all_frames[idx] for idx in sample_indices]

        # Embed each sampled frame
        embedded_list = []
        for frame in sampled_frames:
            # Convert to float in [0,1], shape (H, W, C)
            img_jax = jnp.array(frame) / 255.0
            z = foundation_model.embed_img(img_jax)  # shape (D,)
            embedded_list.append(z)

        # Compute open-endedness score
        if not embedded_list:
            continue
        zs = jnp.stack(embedded_list, axis=0)  # (oe_samples, D)
        oe_score = calc_open_endedness_score(zs)  # scalar

        # Save to dictionary (prompt -> float value)
        oe_scores_per_prompt[prompt] = float(oe_score)

    # Write the results to 'oe_scores.json'
    out_json_path = os.path.join(directory, "oe_scores.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(oe_scores_per_prompt, f, indent=4)
    print(f"Saved open-endedness scores to: {out_json_path}")


if __name__ == "__main__":
    # Root directory containing subdirectories (each with its own config.txt, prompts.txt, etc.)
    root_dir = "data/temporal_large_run"

    # Initialize CLIP (foundation model)
    fm = CLIP()

    # Iterate over each subdirectory in root_dir
    all_subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    
    for subdir_name in all_subdirs:
        subdir_path = os.path.join(root_dir, subdir_name)
        config_file = os.path.join(subdir_path, "config.txt")
        prompts_file = os.path.join(subdir_path, "prompts.txt")
        media_videos_dir = os.path.join(subdir_path, "media", "videos")
        
        # Check that the key files/folders exist
        if (os.path.isfile(config_file)
            and os.path.isfile(prompts_file)
            and os.path.isdir(media_videos_dir)):
            
            # Compute and save the open-endedness scores
            compute_gif_oe_scores(directory=subdir_path, foundation_model=fm)
        else:
            print(f"Skipping '{subdir_name}' — missing config.txt, prompts.txt, or media/videos/")
