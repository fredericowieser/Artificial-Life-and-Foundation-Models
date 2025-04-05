import os
import json
import jax.numpy as jnp

# Import the BGEEmbed text embedding class
# (Adjust the relative import path as needed, for example
#  "from text_embedding import BGEEmbed" if in same directory)
from text_embedding import BGEEmbed

# Reuse the open-endedness metric from your existing pipeline
from asal.asal_metrics import calc_open_endedness_score

def parse_config(config_path):
    """
    Reads a key:value config file (e.g. config.txt) and returns a dict of config parameters.
    Each line is expected to contain 'key: value'.

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

def compute_text_oe_score(directory, embedder):
    """
    Reads prompts from 'prompts.txt' in the given directory, embeds them with
    BGEEmbed, calculates an open-endedness score, and writes it to 'oe_text_scores.json'.

    Parameters
    ----------
    directory : str
        A path to a directory that contains 'prompts.txt' (and optionally config.txt).
    embedder : BGEEmbed
        An instance of the BGEEmbed text embedding class.

    Returns
    -------
    None
        Saves the open-endedness score(s) in 'oe_text_scores.json' within the directory.
    """
    # Parse config (if present), though we might not strictly need it for text-based OE
    config = parse_config(os.path.join(directory, "config.txt"))  # optional usage

    # Load prompts
    prompts_path = os.path.join(directory, "prompts.txt")
    if not os.path.isfile(prompts_path):
        print(f"Warning: No prompts.txt in {directory}, skipping.")
        return

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print(f"No prompts found in {prompts_path}, skipping.")
        return

    # Embed each prompt using BGEEmbed
    # The embed_text(...) method returns a torch.Tensor shape [batch_size, hidden_dim]
    emb_batch = embedder.embed_text(prompts)  # shape: (T, D)

    # Convert to jax numpy array to match the `calc_open_endedness_score` signature
    emb_batch_jax = jnp.array(emb_batch.cpu().numpy())  # shape (T, D)

    # Compute the open-endedness score
    text_oe_score = calc_open_endedness_score(emb_batch_jax)
    text_oe_score_float = float(text_oe_score)

    # Prepare a results dictionary. You could store more info if desired.
    results = {
        "text_oe_score": text_oe_score_float,
        "prompts": prompts
    }

    # Write results to JSON
    output_path = os.path.join(directory, "oe_text_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved text OE score to: {output_path}")

if __name__ == "__main__":
    """
    Example usage:
    1. Create an instance of the text embedding class BGEEmbed.
    2. Loop over each subfolder in 'root_dir'.
    3. For each folder, read prompts.txt -> embed -> compute OE -> save JSON.

    Note: The 'text_embedding.py' file must be in the same package or
          importable path for 'from .text_embedding import BGEEmbed' to work.
    """

    # Top-level directory containing multiple subdirectories
    root_dir = "data/temporal_large_run"

    # Create the text embedder
    text_embedder = BGEEmbed()

    # Iterate over each subdirectory
    subdirectories = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for subdir in subdirectories:
        subdir_path = os.path.join(root_dir, subdir)
        config_file = os.path.join(subdir_path, "config.txt")
        prompts_file = os.path.join(subdir_path, "prompts.txt")

        # Check presence of prompts.txt (and optionally config.txt)
        if os.path.isfile(prompts_file):
            # Compute text-based open-endedness score
            compute_text_oe_score(directory=subdir_path, embedder=text_embedder)
        else:
            print(f"Skipping '{subdir}': missing prompts.txt")
