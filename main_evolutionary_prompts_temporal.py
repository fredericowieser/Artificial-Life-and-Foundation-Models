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
# Gemma 3
from asal_pytorch.foundation_models.gemma3 import Gemma3Chat

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
        return loss_val, dict(
            loss=loss_val,
            loss_prompt=loss_prompt,
            loss_softmax=loss_softmax,
            loss_oe=loss_oe
        )

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

            # Save the given prompt, generated prompt and similarity score
            with open(os.path.join(args.save_dir, "losses.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["loss", "loss_prompt", "loss_softmax", "loss_oe"])
                for d in data_log:
                    writer.writerow([
                        d["best_loss"],
                        d["loss_dict"]["loss_prompt"],
                        d["loss_dict"]["loss_softmax"],
                        d["loss_dict"]["loss_oe"],
                    ])

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
    1) We keep a dynamic list of prompts: all_prompts
    2) On iteration i, we pass the *first i prompts* to the CMA run, so time_sampling=i.
       That means the single rollout is chunked into i segments, each matched to one prompt.
    3) We produce a final video for iteration i, then feed it to Gemma for the next prompt.
    4) Over N iterations, we have i=1..N => 1 + 2 + ... + N total prompts appended across iterations.
    """

    gemma = Gemma3Chat()

    splitted = args.prompts.split(";")
    if len(splitted) < 1:
        splitted = ["a default prompt"]
    # Start with just the first prompt
    all_prompts = [splitted[0]]

    rng = None
    current_params = None
    final_video_paths = []

    for i in range(1, args.N + 1):
        print(f"\n=== Starting iteration {i} ===")
        # We have i prompts so far: all_prompts[:i]
        # We'll run 1 CMA-ES loop with time_sampling = i
        # i.e. each chunk in the same rollout matches one prompt
        prompts_for_i = all_prompts[:i]
        best_params, video_frames, rng = run_for_iteration(
            args,
            rng=rng,
            iteration_idx=i,
            prompt_list=prompts_for_i,
            init_params=current_params
        )

        # Save final iteration video
        if args.save_dir:
            video_path_i = os.path.join(args.save_dir, f"video_iteration_{i}.mp4")
            imageio.mimsave(video_path_i, video_frames, fps=30, codec="libx264")
            print(f"[Iteration {i}] final video saved at: {video_path_i}")
            final_video_paths.append(video_path_i)

        # Show final video to Gemma => get new prompt
        instruction =(f"You just saw the video for iteration {i}, which used prompts so far: {all_prompts}. "
            "Suggest a NEW single prompt (no extra text) to expand upon this evolution next time."
        )
        new_prompt = gemma.describe_video(
            video_frames,
            extract_prompt=instruction,
            max_tokens=20,
            temperature=args.temp,
            max_images=args.max_images,
        )
        print(f"[Iteration {i}] Gemma suggested => '{new_prompt}'")

        # Append new prompt to our list => next iteration will have i+1 total prompts
        all_prompts.append(new_prompt)

        # Our final best_params becomes the init for next iteration
        current_params = best_params
        if args.save_dir:
            save_final_prompts_csv(all_prompts, args.save_dir)

        # Save the given prompt, generated prompt and similarity score
        with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
            # Write all args
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")


if __name__ == '__main__':
    main(parse_args())
