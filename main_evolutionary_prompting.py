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
import wandb

import asal.substrates as substrates
import asal.foundation_models as foundation_models
from asal.rollout import rollout_simulation
import asal.asal_metrics as asal_metrics
import asal.util as util
from asal.wandb_util import WandbLogger

from asal_pytorch.foundation_models.gemma3 import Gemma3Chat

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")
group.add_argument("--wandb", action="store_true", help="log to wandb; default false unless flag is given")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='boids', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, leave None for the default of the substrate")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use (don't touch this)")
group.add_argument("--time_sampling", type=int, default=1, help="number of images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells", help="prompts to optimize for seperated by ';'")
group.add_argument("--coef_prompt", type=float, default=1., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss (only for multiple temporal prompts)")
group.add_argument("--coef_oe", type=float, default=0., help="coefficient for ASAL open-endedness loss (only for single prompt)")

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=16, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--N", type=int, default=3, help="num_of_loops")

def load_best_params(save_dir):
    """Load the best parameters from the saved pickle file."""
    best_params_path = os.path.join(save_dir, "best.pkl")
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
    return best_params[0]  # Extract best parameters
def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def run_optimisation(args,rng, iteration=0, wandb_logger=None):
    prompts = args.prompts.split(";")
    if args.time_sampling < len(prompts): # doing multiple prompts
        args.time_sampling = len(prompts)
    print(args)
    
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=args.rollout_steps, time_sampling=(args.time_sampling, True), img_size=224, return_state=False)

    z_txt = fm.embed_txt(prompts) # P D

    # Initialise wandb logging
    if args.wandb:
        wandb_logger.initialise_rollout(substrate=substrate)

    rng = jax.random.PRNGKey(args.seed)
    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
    es_params = strategy.default_params
    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params): # calculate the loss given the simulation parameters
        rollout_data = rollout_fn(rng, params)
        z = rollout_data['z']

        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z)

        loss = loss_prompt * args.coef_prompt + loss_softmax * args.coef_softmax + loss_oe * args.coef_oe
        loss_dict = dict(loss=loss, loss_prompt=loss_prompt, loss_softmax=loss_softmax, loss_oe=loss_oe)
        return loss, loss_dict

    @jax.jit
    def do_iter(es_state, rng): # do one iteration of the optimization
        rng, _rng = split(rng)
        params, next_es_state = strategy.ask(_rng, es_state, es_params)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0)) # vmap over the init state rng and then the parameters
        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict)) # mean over the init state rng
        next_es_state = strategy.tell(params, loss, next_es_state, es_params)
        data = dict(best_loss=next_es_state.best_fitness, loss_dict=loss_dict)
        return next_es_state, data

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)

        data.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())
        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1): # save data every 10% of the run
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best)

        # Wandb logging
        if args.wandb:
            wandb_logger.log_losses(di)

    if args.wandb:
        params, _ = util.load_pkl(args.save_dir, "best")
        rng = jax.random.PRNGKey(args.seed)
        wandb_logger.log_video(rng, params, f"iteration_{iteration}")
        print(f"Iteration {iteration} video logged to wandb")

    if args.save_dir is not None: 
        best_params = load_best_params(args.save_dir)
        # Using The Best Found Parameters
        rollout_fn = partial(
            rollout_simulation,
            s0=None,
            substrate=substrate,
            fm=None,
            rollout_steps=args.rollout_steps,
            time_sampling='video',
            img_size=224,
            return_state=False,
        )

        # Run simulation with best parameters
        rollout_data = rollout_fn(rng, best_params)

        # Convert frames from float [0,1] to uint8 [0,255] for video
        rgb_frames = np.array(rollout_data['rgb'])
        video_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)
        
        video_path = os.path.join(args.save_dir, f"video_{iteration}.mp4")
        imageio.mimsave(video_path, video_frames, fps=30, codec="libx264")
        print(f"Video saved at: {video_path}")

        return video_path, video_frames, rng
    
def main(args):

    if args.save_dir is not None:
        prompt_file = os.path.join(args.save_dir, "evolved_prompts.txt")

    # Initialise wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(project="alife-project", group="evolutionary-prompting", entity="ucl-asal", config=vars(args))
        wandb_logger.initialise_prompt_logging()
        wandb_logger.log_prompt(args.prompts)
    else:
        wandb_logger = None

    final_video_paths = []
    final_frames = []  # To collect frames from all iterations

    # Initial ASAL run (iteration 0)
    video_path, video_frames, rng = run_optimisation(args, rng=None, iteration=0, wandb_logger=wandb_logger)
    final_video_paths.append(video_path)
    final_frames.extend(video_frames)
    current_prompt = args.prompts  # starting prompt

    # Initialize Gemma3Chat for feedback.
    gemma = Gemma3Chat()

    for i in range(args.N):
        print(f"Gemma iteration {i+1}/{args.N}")
        
        evolve_instruction = (
            f"""This artificial life simulation was optimised to produce PREVIOUS TARGET PROMPT: '{current_prompt}'.

            Your task is to provide a NEXT TARGET PROMPT for the next stage of the artificial life evolution, following on from the previous prompt and simulation. Your aim is to create a diverse, interesting and meaningfully different life form. Use your imagination, but keep your target prompt simple and concise. ONLY output the new target prompt and nothing else.

            NEXT TARGET PROMPT: """
        )

        evolved_prompt=gemma.describe_video(video_frames,extract_prompt=evolve_instruction, max_tokens=15)

        if args.save_dir is not None:
            # Log prompt file to save_dir
            with open(prompt_file, "a") as f:
                f.write(f"Iteration {i+1}: {evolved_prompt}\n")
            
        # Log the prompt file and text to wandb
        if args.wandb:
            wandb_logger.log_prompt(evolved_prompt)       

        print("Gemma suggested new prompt", evolved_prompt)
        current_prompt=evolved_prompt
        args.prompts = evolved_prompt

        video_path, video_frames, rng = run_optimisation(args, rng, iteration=i+1, wandb_logger=wandb_logger)
        final_video_paths.append(video_path)
        final_frames.extend(video_frames)
    
    if args.save_dir is not None :
        final_video_path=os.path.join(args.save_dir, "final_video.mp4")
        imageio.mimsave(final_video_path, final_frames, fps=30,codec="libx264" )
        print(f"Final video saved at: {final_video_path}")


if __name__ == '__main__':
    main(parse_args())


