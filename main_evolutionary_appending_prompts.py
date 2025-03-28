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

# Gemma 3
from asal_pytorch.foundation_models.gemma3 import Gemma3Chat

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='boids', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, or None for default")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--time_sampling", type=int, default=1, help="images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells",
                   help="the initial prompts, we only use the first for iteration 1")
group.add_argument("--coef_prompt", type=float, default=1., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss")
group.add_argument("--coef_oe", type=float, default=0., help="coefficient for open-endedness loss")

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=16, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations per block")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--N", type=int, default=3, help="total number of iterations (videos) we want")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args

def load_best_params(save_dir):
    """Load the best parameters (best_member) from best.pkl."""
    best_path = os.path.join(save_dir, "best.pkl")
    with open(best_path, "rb") as f:
        data = pickle.load(f)
    # data[0] => best_member, data[1] => best_fitness
    return data[0]

def run_block_optimisation(
    args,
    rng,
    block_label,
    prompt,
    init_params=None,
    n_iters=1000
):
    """
    Runs a single "block" of CMA-ES with one prompt for n_iters steps.
    If init_params is None, we initialize CMA-ES from scratch.
    Otherwise, we seed CMA-ES with init_params.
    Returns (best_params, video_frames, rng).
    Also saves best.pkl inside args.save_dir (overwriting older best.pkl).
    """
    print(f"\n=== Block '{block_label}' with prompt '{prompt}' for {n_iters} iterations ===")

    # 1) Prepare substrate & foundation model
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    # 2) Build rollout function
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling=(args.time_sampling, True),
        img_size=224,
        return_state=False
    )

    # We'll pass just a single prompt to this block
    z_txt = fm.embed_txt([prompt])  # shape: (1, D)

    # 3) CMA-ES setup
    if rng is None:
        rng = jax.random.PRNGKey(args.seed)

    strategy = evosax.Sep_CMA_ES(
        popsize=args.pop_size,
        num_dims=substrate.n_params,
        sigma_init=args.sigma
    )
    es_params = strategy.default_params
    rng, init_rng = split(rng)
    es_state = strategy.initialize(init_rng, es_params)

    # Now override the mean in the state, if you have init_params:
    if init_params is not None:
        # If EvoState is a FLAX struct, you can do:
        es_state = es_state.replace(mean=init_params)

    # If that fails with “no replace()”, but you see "_replace" in dir(es_state),
    # it’s a namedtuple-like object. Then do:
    # es_state = es_state._replace(mean=init_params)

    print("es_state =", es_state)
    print("dir(es_state) =", dir(es_state))


    # if init_params is not None:
    #     state_dict = unfreeze(es_state)
    #     # Update the field that stores the current search mean.
    #     # Check the available keys by printing: print(state_dict.keys())
    #     # Commonly, it is "mean". If not, use the key that holds your solution.
    #     state_dict["mean"] = init_params  
    #     es_state = freeze(state_dict)

    

    def calc_loss(rng_in, candidate_params):
        rollout_data = rollout_fn(rng_in, candidate_params)
        z = rollout_data['z']

        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z)

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

        # "tell" CMA-ES the fitness
        next_es_state = strategy.tell(params, losses, next_es_state, es_params)

        out_data = {
            'best_loss': next_es_state.best_fitness,
            'loss_dict': loss_dicts
        }
        return next_es_state, out_data, rng_inner

    # 4) Main loop for n_iters
    data_log = []
    pbar = tqdm(range(n_iters), desc=f"Block {block_label}")
    for i in pbar:
        es_state, di, rng = do_iter(es_state, rng)
        data_log.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())

        # Save occasionally
        if args.save_dir is not None and (i == n_iters-1 or i % max(1, n_iters//10) == 0):
            data_save = jax.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data_log)
            util.save_pkl(args.save_dir, "data", data_save)

            best_blob = jax.tree_map(lambda x: np.array(x),
                                     (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best_blob)

    # 5) Load final best params from disk
    if args.save_dir:
        best_params = load_best_params(args.save_dir)
    else:
        best_params = es_state.best_member

    # 6) Produce final frames from these best params
    rollout_fn_video = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=None,
        rollout_steps=args.rollout_steps,
        time_sampling='video',
        img_size=224,
        return_state=False
    )
    rollout_data = rollout_fn_video(rng, best_params)
    video_frames = (np.array(rollout_data['rgb']) * 255).clip(0, 255).astype(np.uint8)

    return best_params, video_frames, rng

def main(args):
    """
    We have N total iterations (i = 1..N).  For iteration i:

    - We have i prompts so far in the list all_prompts.
    - We chain i blocks of 1000 steps (args.n_iters each):
      block 1 uses prompt #1, block 2 uses prompt #2, ..., block i uses prompt #i.
      each block starts from the best params of the previous block.
    - We produce 1 final video at the end of iteration i (i.e., after block i).
    - Gemma sees that video => suggests new prompt => appended to all_prompts.
    - So iteration i performs i * 1000 steps. Summing i=1..N => total 1000(1+2+...+N).

    That yields exactly N final videos.
    """

    # 1) Set up Gemma
    gemma = Gemma3Chat()

    # 2) We store all prompts in a list. The first one is from args.prompts
    #    (the user might have typed multiple prompts separated by ';',
    #    but we only use the first as the "original" or "prompt #1")
    splitted = args.prompts.split(";")
    if len(splitted) < 1:
        splitted = ["a default prompt"]
    all_prompts = [splitted[0]]  # the "original" prompt

    rng = None  # We'll let the first call create a PRNG from args.seed
    current_params = None

    # This is where we store each iteration’s final video path
    final_video_paths = []

    # ============ ITERATIONS i = 1..N ============
    for i in range(1, args.N + 1):
        print(f"\n======== Starting Iteration {i} ========")
        # We have i prompts so far: all_prompts[0..i-1]
        # We'll do i blocks, each 1000 steps.

        # Start each iteration with the best params from the end of the previous iteration
        iteration_params = current_params
        iteration_frames = None

        for block_index in range(i):
            prompt_i = all_prompts[block_index]  # block_index from 0..(i-1)
            block_label = f"iter{i}_block{block_index+1}"
            iteration_params, iteration_frames, rng = run_block_optimisation(
                args,
                rng=rng,
                block_label=block_label,
                prompt=prompt_i,
                init_params=iteration_params,
                n_iters=args.n_iters
            )
            # Now iteration_params is best from that block

        # After i blocks => iteration_params is best from the last block
        # iteration_frames are the frames from that last block's best
        # Save final iteration video
        video_path_i = None
        if args.save_dir:
            video_path_i = os.path.join(args.save_dir, f"video_iteration_{i}.mp4")
            imageio.mimsave(video_path_i, iteration_frames, fps=30, codec="libx264")
            print(f"[Iteration {i}] final video saved at: {video_path_i}")
            final_video_paths.append(video_path_i)

        # Gemma sees the final video => suggests the next new prompt
        gemma_instruction = (
            f"You just saw the video for iteration {i}, which used prompts so far: {all_prompts}. "
            "Suggest a NEW single prompt (no extra text) to expand upon this evolution next time."
        )
        if iteration_frames is None:
            # If we didn't have frames, skip Gemma in this example
            new_prompt = "No frames available"
        else:
            new_prompt = gemma.describe_video(
                iteration_frames,
                extract_prompt=gemma_instruction,
                max_tokens=20
            )
        print(f"[Iteration {i}] Gemma’s new prompt => '{new_prompt}'")

        # Append this new prompt to all_prompts => so iteration i+1 has i+1 prompts
        all_prompts.append(new_prompt)

        # The final best params for iteration i => used as start next time
        current_params = iteration_params

    # If you want to combine everything into one giant final video, you can read
    # the videos again or store frames along the way. For now, we have N videos.

if __name__ == '__main__':
    main(parse_args())
