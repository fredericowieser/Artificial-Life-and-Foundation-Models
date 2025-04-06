import os
import re
import imageio
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def parse_config(config_path):
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    config[key.strip()] = val.strip()
    return config

def create_prompt_gif_figure(directory, N=5, output_filename="prompt_gif_figure.png"):
    config = parse_config(os.path.join(directory, "config.txt"))
    main_prompt = config.get("prompts", "Unknown Prompt")
    n_iters = config.get("n_iters", "?")

    with open(os.path.join(directory, "prompts.txt"), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    media_dir = os.path.join(directory, "media", "videos")
    gif_files = [f for f in os.listdir(media_dir) if f.endswith(".gif")]
    iteration_to_gif = {}
    for fname in gif_files:
        match = re.match(r"^iteration_(\d+)_.*\.gif$", fname)
        if match:
            idx = int(match.group(1)) - 1
            iteration_to_gif[idx] = os.path.join(media_dir, fname)

    fig = plt.figure(figsize=(N * 2, len(prompts) * 3), dpi=150)
    gs = GridSpec(len(prompts), 1, figure=fig, hspace=0.03)

    for i, prompt in enumerate(prompts):
        if i not in iteration_to_gif:
            continue

        frames = imageio.mimread(iteration_to_gif[i])
        if not frames:
            continue

        sample_indices = np.linspace(0, len(frames) - 1, N, dtype=int)
        first = frames[0]
        frame_h, frame_w = first.shape[:2]
        total_w = frame_w * N
        total_h = frame_h

        ax_main = fig.add_subplot(gs[i, 0])
        ax_main.set_xlim(0, total_w)
        ax_main.set_ylim(0, total_h)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_aspect("equal")
        ax_main.set_facecolor("black")
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        for j, idx in enumerate(sample_indices):
            img = frames[idx]
            x0 = j * frame_w
            x1 = (j + 1) * frame_w
            ax_main.imshow(img, extent=[x0, x1, 0, frame_h], origin="lower")

        # Draw white vertical lines between frames
        for j in range(1, N):  # skip first (no line at left edge)
            x = j * frame_w
            ax_main.axvline(x=x, color='white', linewidth=1)

        # Create the prompt bar above the main axis using AxesDivider
        divider = make_axes_locatable(ax_main)
        ax_bar = divider.append_axes("top", size="25%", pad=0.01, sharex=ax_main)
        ax_bar.set_xlim(0, total_w)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_facecolor("black")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        ax_bar.text(
            total_w / 2, 0.5,
            f"\"{prompt}\"",
            ha="center", va="center", color="white", fontsize=25
        )

    fig.suptitle(f"Temporally Evolved Supervised Target", fontsize=30, color="black", y=0.9)
    out_path = os.path.join(directory, output_filename)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved figure to: {out_path}")

if __name__ == "__main__":
    """
    In this block, we'll iterate over each subfolder in data/temporal_large_run/
    and run the figure-generation function if it has config.txt, prompts.txt,
    and a media/videos folder.
    """
    root_dir = "data/temporal_large_run"
    
    # List all subdirectories in root_dir
    all_subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    
    for subdir_name in all_subdirs:
        subdir_path = os.path.join(root_dir, subdir_name)
        
        # Check existence of key files/folders
        config_file = os.path.join(subdir_path, "config.txt")
        prompts_file = os.path.join(subdir_path, "prompts.txt")
        media_videos_dir = os.path.join(subdir_path, "media", "videos")
        
        if (os.path.isfile(config_file)
            and os.path.isfile(prompts_file)
            and os.path.isdir(media_videos_dir)):
            
            # Construct an output figure name like "<subdir_name>_prompt_gif_figure.png"
            out_filename = f"{subdir_name}_prompt_gif_figure.pdf"
            
            # Run the creation function
            create_prompt_gif_figure(
                directory=subdir_path,
                N=5,  # how many frames per GIF
                output_filename=out_filename
            )
        else:
            print(f"\nSkipping '{subdir_name}' — missing config.txt, prompts.txt, or media/videos/")
