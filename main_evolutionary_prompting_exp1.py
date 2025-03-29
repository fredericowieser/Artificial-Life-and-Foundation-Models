import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial
import pickle
import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm
import imageio
import asal.substrates as substrates
import asal.foundation_models as foundation_models
from asal.rollout import rollout_simulation
import asal.asal_metrics as asal_metrics
import asal.util as util
import csv
import re
import shutil
import ffmpeg

import glob
import tempfile
import shutil
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Gemma 3
from asal_pytorch.foundation_models.gemma3 import Gemma3Chat
def sanitize_filename(s):
    s = s.strip().lower()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-]', '', s)
    return s[:50]  # limit length
parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='boids', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="foundation model to use")
group.add_argument("--time_sampling", type=int, default=1, 
                   help="images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells",
                   help="the initial prompts (we only use the first as the 'original' prompt #1)")
group.add_argument("--coef_prompt", type=float, default=1., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0.,
                   help="coefficient for softmax loss (used for multiple temporal prompts)")
group.add_argument("--coef_oe", type=float, default=0., help="coefficient for open-endedness loss")

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=16, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="CMA-ES steps per iteration")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--N", type=int, default=3, help="total number of Gemma loops")
group.add_argument("--temp", type=float, default=0.0, help="Temperature for sampling")
group.add_argument("--max_images", type=int, default=10, help="Number of image to give to Foundation Model")
group.add_argument("--S", type=int, default=1, help="number of child branches to spawn per branch")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args

def load_best_params(save_dir):
    """Load best member from best.pkl."""
    best_path = os.path.join(save_dir, "best.pkl")
    with open(best_path, "rb") as f:
        data = pickle.load(f)
    # data[0] => best_member, data[1] => best_fitness
    return data[0]
def get_unique_prompts(gemma,video_frames,instruction,S,max_retries=10):
    unique_prompts=[]
    retries=0
    while len(unique_prompts)<S and retries<max_retries:
        new_prompt=gemma.describe_video(
            video_frames,
            extract_prompt=instruction,
            max_tokens=20,
        ).strip()
        norm_prompt=new_prompt.lower()
        if norm_prompt not in [p.lower() for p in unique_prompts]:
            unique_prompts.append(new_prompt)
        else:
            print(f"Duplicate prompt detected: '{new_prompt}'. Retrying...")
        retries+=1
    if len(unique_prompts)<S:
        print("Warning: Could not generate S unique prompts after max retries.")
    return unique_prompts


def run_for_iteration(
    args,
    rng,
    iteration_idx,
    prompt_list,          # all prompts for iteration i
    init_params=None
):
    """
   We do time_sampling = len(prompt_list).
    That means each time chunk in the same rollout is matched to a different prompt.
    Returns best_params, video_frames, updated rng.
    """
    print(f"\n=== Iteration {iteration_idx} with prompts:", prompt_list, "===")

    # 1) Prepare foundation model & substrate
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps
    S=args.S
    # 2) We'll have as many prompts as we have time chunks
    n_prompts = len(prompt_list)
    # We'll override the user's time_sampling for the rollout,
    # so that each chunk can be matched to one prompt.
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling=(n_prompts, True),  # e.g. (2, True) if 2 prompts
        img_size=224,
        return_state=False
    )

    # embed all prompts at once => shape (P, D), where P = n_prompts
    z_txt = fm.embed_txt(prompt_list)

    # 3) CMA-ES setup
    if rng is None:
        rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = split(rng)

    strategy = evosax.Sep_CMA_ES(
        popsize=args.pop_size,
        num_dims=substrate.n_params,
        sigma_init=args.sigma
    )
    es_params = strategy.default_params
    es_state = strategy.initialize(init_rng, es_params)

    # If continuing from previous iteration, set the state's mean to init_params
    if init_params is not None:
        es_state = es_state.replace(mean=init_params)

    # define the CMA-ES loss
    def calc_loss(rng_in, candidate_params):
        rollout_data = rollout_fn(rng_in, candidate_params)
        z = rollout_data['z']  # shape: (n_prompts, embedding_dim)

        # For multiple prompts, we measure how each time chunk's embedding matches
        # its corresponding prompt in z_txt. Inside asal, we can do e.g.:
        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z)

        # Weighted sum
        loss_val = (loss_prompt * args.coef_prompt
                    + loss_softmax * args.coef_softmax
                    + loss_oe * args.coef_oe)
        return loss_val, {
            'loss': loss_val,
            'loss_prompt': loss_prompt,
            'loss_softmax': loss_softmax,
            'loss_oe': loss_oe
        }

    @jax.jit
    def do_iter(es_state_inner, rng_inner):
        rng_inner, ask_rng = split(rng_inner)
        params, next_es_state = strategy.ask(ask_rng, es_state_inner, es_params)

        calc_loss_vv = jax.vmap(
            jax.vmap(calc_loss, in_axes=(0, None)),
            in_axes=(None, 0)
        )
        rng_inner, batch_rng = split(rng_inner)
        losses, loss_dicts = calc_loss_vv(split(batch_rng, args.bs), params)
        # Average across the 'bs' dimension
        losses, loss_dicts = jax.tree_map(lambda x: x.mean(axis=1), (losses, loss_dicts))

        next_es_state = strategy.tell(params, losses, next_es_state, es_params)
        data_out = {
            'best_loss': next_es_state.best_fitness,
            'loss_dict': loss_dicts
        }
        return next_es_state, data_out, rng_inner

    # 4) CMA-ES main loop
    data_log = []
    pbar = tqdm(range(args.n_iters), desc=f"Iteration {iteration_idx}")
    for i in pbar:
        es_state, di, rng = do_iter(es_state, rng)
        data_log.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())

        # occasionally save
        if args.save_dir is not None and (i == args.n_iters - 1 or i % max(1, args.n_iters // 10) == 0):
            data_save = jax.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data_log)
            util.save_pkl(args.save_dir, "data", data_save)

            best_blob = jax.tree_map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best_blob)

    # after done with n_iters, load the best params from disk
    if args.save_dir:
        best_params = load_best_params(args.save_dir)
    else:
        best_params = es_state.best_member

    # produce final frames from best params
    rollout_fn_video = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=None,  # no need to embed again
        rollout_steps=args.rollout_steps,
        time_sampling='video',
        img_size=224,
        return_state=False
    )
    rollout_data = rollout_fn_video(rng, best_params)
    rgb = np.array(rollout_data['rgb'])
    video_frames = (rgb * 255).clip(0, 255).astype(np.uint8)

    return best_params, video_frames, rng
def save_final_prompts_csv(all_prompts, folder):
    """Saves the list of prompts to 'final_prompts.csv' in the given folder."""
    csv_path = os.path.join(folder, "final_prompts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt"])
        for p in all_prompts:
            writer.writerow([p])
    print(f"Final prompts saved at: {csv_path}")



    
def main(args):
    """
    Branching experiment:
      - Start with an initial branch (with one initial prompt).
      - At each meta-iteration, each branch evolves using its prompt chain.
      - After evolution, Gemma is used to produce S unique new prompts for each branch.
      - For each new prompt, a child branch is created by appending the prompt to the parent's chain.
      - Videos are saved in a tree-structured folder corresponding to each branch.
      - Final prompt chains for all branches are saved in a CSV.
    """
    gemma = Gemma3Chat()

    splitted = args.prompts.split(";")
    if len(splitted) < 1:
        splitted = ["a default prompt"]
    # Start with just the first prompt
    initial_prompt = splitted[0]
    
    # Initialize one branch.
    branches = [{
        "prompt_chain": [initial_prompt],
        "current_params": None,
        "rng": None,
        "folder_path": [sanitize_filename(initial_prompt)]
    }]

    base_save_dir=args.save_dir if args.save_dir is not None else "results"
    os.makedirs(base_save_dir, exist_ok=True)

    S=args.S
    meta_iterations = args.N

    for meta in range(1, meta_iterations + 1):
        print(f"\n--- Meta Iteration {meta} ---")
        new_branches = []
        for branch_index, branch in enumerate(branches):
            prompt_chain = branch["prompt_chain"]
            rng = branch["rng"]
            current_params = branch["current_params"]
            
            # Build folder for this branch.
            branch_folder = os.path.join(base_save_dir, *branch["folder_path"])
            os.makedirs(branch_folder, exist_ok=True)
            
            # Run CMA-ES for this branch.
            best_params, video_frames, updated_rng =run_for_iteration(
                args,
                rng=rng,
                iteration_idx=meta,
                prompt_list=prompt_chain,
                init_params=current_params
            )
            
            # Save final video for this branch.
            video_path = os.path.join(branch_folder, f"video_meta_{meta}.mp4")
            imageio.mimsave(video_path, video_frames, fps=30, codec="libx264")
            print(f"Saved video for branch {branch_index} at meta {meta}: {video_path}")
            
            # Save prompt chain as a text file.
            chain_txt_path = os.path.join(branch_folder, "prompt_chain.txt")
            with open(chain_txt_path, "w", encoding="utf-8") as f:
                f.write(" ; ".join(prompt_chain))
            
            # Create an instruction for Gemma.
            instruction = (f"This video was produced using the prompt chain: {prompt_chain}. "
                           "Propose a NEW single prompt (no extra text) to further evolve this branch.")
            
            # Get S unique new prompts for this branch.
            unique_child_prompts = get_unique_prompts(gemma, video_frames, instruction, S)
            print(f"Unique prompts for branch {branch_index} at meta {meta}:", unique_child_prompts)
            
            # Create child branches.
            for child_prompt in unique_child_prompts:
                child_prompt_chain = prompt_chain + [child_prompt]
                child_folder_name = sanitize_filename(child_prompt)
                child_folder_path = branch["folder_path"] + [child_folder_name]
                new_branches.append({
                    "prompt_chain": child_prompt_chain,
                    "current_params": best_params,
                    "rng": updated_rng,
                    "folder_path": child_folder_path
                })
        branches = new_branches
    
    # Save final prompt chains from all final branches into a CSV.
    final_csv_path = os.path.join(base_save_dir, "final_prompt_chains.csv")
    with open(final_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["branch_index", "prompt_chain"])
        for idx, branch in enumerate(branches):
            writer.writerow([idx, " ; ".join(branch["prompt_chain"])])
    print(f"Final prompt chains saved at: {final_csv_path}")
    
    # Create a final summary video that shows all branch videos.
    # We assume that each final branch video is the one produced at the last meta iteration.
    


    def overlay_text_pillow(image_path, text_str, out_path, font_path=None, font_size=24):
        """
        Loads 'image_path', draws 'text_str' onto it near the bottom center,
        and saves as 'out_path'.
        """
        # 1. Open the image.
        im = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(im)

        # 2. Load a font.
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # 3. Measure text using font.getsize(...).
        text_w, text_h = font.getsize(text_str)

        # 4. Decide where to place the text (e.g. bottom-center).
        x = (im.width - text_w) // 2
        y = im.height - text_h - 20  # 20px from bottom

        # 5. Draw the text in white, for example.
        draw.text((x, y), text_str, font=font, fill=(255, 255, 255))

        # 6. Save output image.
        im.save(out_path)
        print(f"Saved labeled image to: {out_path}")

    def create_final_summary_video_pillow(base_save_dir, final_meta, output_path="final_summary.mp4", fps=30,
                                        font_path=None, margin_bottom=50, font_size=24):
        """
        1) Finds all branch videos named 'video_meta_{final_meta}.mp4' in base_save_dir.
        2) For each, reads the prompt chain from 'prompt_chain.txt' in the same folder.
        3) Overlays text on each video using PIL (frame-by-frame).
        4) Concatenates all labeled mp4 files into a single final mp4 via ffmpeg concat.
        """
        pattern = os.path.join(base_save_dir, '**', f"video_meta_{final_meta}.mp4")
        video_paths = glob.glob(pattern, recursive=True)
        if not video_paths:
            print("No final branch videos found for meta iteration:", final_meta)
            return

        temp_dir = tempfile.mkdtemp(prefix="labeled_videos_")
        labeled_files = []

        # 1) Label each final video with PIL
        for idx, video_path in enumerate(sorted(video_paths)):
            branch_dir = os.path.dirname(video_path)
            chain_path = os.path.join(branch_dir, "prompt_chain.txt")
            if os.path.exists(chain_path):
                with open(chain_path, "r", encoding="utf-8") as f:
                    prompt_chain = f.read().strip().split(" ; ")
            else:
                prompt_chain = []

            labeled_output = os.path.join(temp_dir, f"labeled_{idx}.mp4")
            print(f"Overlaying text on {video_path} with prompts: {prompt_chain}")
            overlay_text_pillow(
                input_video=video_path,
                prompt_chain=prompt_chain,
                output_video=labeled_output,
                fps=fps,
                font_path=font_path,
                margin_bottom=margin_bottom,
                font_size=font_size
            )
            labeled_files.append(labeled_output)

        # 2) Concat all labeled videos
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for file in labeled_files:
                f.write(f"file '{os.path.abspath(file)}'\n")

        try:
            (
                ffmpeg
                .input(concat_file, format='concat', safe=0)
                .output(output_path, codec='libx264', pix_fmt='yuv420p', r=fps)
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"Final summary video saved at: {output_path}")
        except ffmpeg.Error as e:
            print("FFmpeg concat error:", e)
            raise
        finally:
            # clean up
            shutil.rmtree(temp_dir)
    
    final_meta = args.N  # or however many meta iterations you used
    summary_video_path = os.path.join(args.save_dir, "final_summary.mp4")
    create_final_summary_video_pillow(
        base_save_dir=args.save_dir,
        final_meta=final_meta,
        output_path=summary_video_path,
        fps=30,
        font_path=None,        # or "/path/to/your_font.ttf"
        margin_bottom=100,     # increase to ensure text isn't cut off
        font_size=24
    )


if __name__ == '__main__':
    main(parse_args())