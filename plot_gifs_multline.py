import os
import re
import textwrap
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

matplotlib.use("Agg")  # or omit if interactive

def parse_config(config_path):
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    config[key.strip()] = val.strip()
    return config

def shrink_text_to_fit(ax, text_obj, margin=0.95, min_fontsize=6):
    """
    Iteratively measures the text bounding box vs. the axis 
    and shrinks the font if it's too wide.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    while True:
        bbox_text = text_obj.get_window_extent(renderer=renderer)
        bbox_ax   = ax.get_window_extent(renderer=renderer)
        if (bbox_text.width <= bbox_ax.width * margin) or (text_obj.get_fontsize() <= min_fontsize):
            break
        text_obj.set_fontsize(text_obj.get_fontsize() - 1)
        fig.canvas.draw()

def create_prompt_gif_figure(directory, N=5, output_filename="prompt_gif_figure.pdf"):
    # --- Read config/prompts
    config = parse_config(os.path.join(directory, "config.txt"))
    with open(os.path.join(directory, "prompts.txt"), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # --- Map iteration index -> .gif
    media_dir = os.path.join(directory, "media", "videos")
    gif_files = [f for f in os.listdir(media_dir) if f.endswith(".gif")]
    iteration_to_gif = {}
    for fname in gif_files:
        match = re.match(r"^iteration_(\d+)_.*\.gif$", fname)
        if match:
            idx = int(match.group(1)) - 1
            iteration_to_gif[idx] = os.path.join(media_dir, fname)

    # --- Determine how many lines each prompt needs
    line_counts = []
    for prompt in prompts:
        wrapped = textwrap.fill(prompt, width=50, break_long_words=True)
        lc = wrapped.count('\n') + 1
        line_counts.append(lc)

    # Each row: 3 "units" for GIF frames + line_counts[i] for black bar
    row_heights = [3 + lc for lc in line_counts]

    # Create figure sized for N frames wide, sum(row_heights)+1 tall
    fig_width  = N * 2
    fig_height = sum(row_heights) + 1
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

    # The top-level GridSpec: we use hspace>0 here to separate each row
    #   so there's some vertical gap between one iteration's row and the next.
    gs_top = GridSpec(
        len(prompts), 1,
        figure=fig,
        height_ratios=row_heights,
        hspace=0.5  # <--- spacing BETWEEN different prompt rows
    )

    # Adjust the top so the suptitle is not so far above
    fig.subplots_adjust(top=0.85)

    # Suptitle a bit lower (y=0.93 for example)
    fig.suptitle(
        "Temporally Evolved Supervised Target",
        fontsize=30,
        color="black",
        y=0.88
    )

    # For each prompt row, sub-GridSpec with 2 rows: [bar, frames]
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

        # Sub-grid: top= bar, bottom= frames, with hspace=0 => flush
        sub_gs = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs_top[i],
            height_ratios=[line_counts[i], 3],
            hspace=0.0  # <--- no gap inside each row 
        )

        # ========== Top axis = black bar (flush with frames) ==========
        ax_bar = fig.add_subplot(sub_gs[0])
        ax_bar.set_facecolor("black")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        wrapped_prompt = textwrap.fill(prompt, width=50, break_long_words=True)
        text_obj = ax_bar.text(
            0.5, 0.5,
            f"\"{wrapped_prompt}\"",
            ha="center", va="center",
            color="white", fontsize=25,
            transform=ax_bar.transAxes
        )
        fig.canvas.draw()
        shrink_text_to_fit(ax_bar, text_obj, margin=0.98, min_fontsize=8)

        # ========== Bottom axis = GIF frames ==========
        ax_main = fig.add_subplot(sub_gs[1])
        ax_main.set_facecolor("black")
        ax_main.set_xlim(0, total_w)
        ax_main.set_ylim(0, total_h)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        # Draw each sampled frame side-by-side
        for j, idx in enumerate(sample_indices):
            img = frames[idx]
            x0 = j * frame_w
            x1 = (j + 1) * frame_w
            ax_main.imshow(img, extent=[x0, x1, 0, frame_h], origin="lower")

        # White vertical lines between frames
        for j in range(1, N):
            ax_main.axvline(x=j*frame_w, color='white', linewidth=1)

    out_path = os.path.join(directory, output_filename)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved figure to: {out_path}")

if __name__ == "__main__":
    root_dir = "data/temporal_large_run"
    all_subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    
    for subdir_name in all_subdirs:
        subdir_path = os.path.join(root_dir, subdir_name)
        
        config_file = os.path.join(subdir_path, "config.txt")
        prompts_file = os.path.join(subdir_path, "prompts.txt")
        media_videos_dir = os.path.join(subdir_path, "media", "videos")
        
        if (os.path.isfile(config_file)
            and os.path.isfile(prompts_file)
            and os.path.isdir(media_videos_dir)):
            
            out_filename = f"{subdir_name}_prompt_gif_figure_multi.pdf"
            create_prompt_gif_figure(
                directory=subdir_path,
                N=5,
                output_filename=out_filename
            )
        else:
            print(f"Skipping '{subdir_name}' — missing required files/folders.")
