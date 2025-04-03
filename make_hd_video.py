import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import pickle
from functools import partial

import evosax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import split
from tqdm.auto import tqdm
from einops import repeat

import asal.asal_metrics as asal_metrics
import asal.foundation_models as foundation_models
import asal.substrates as substrates
import asal.util as util
from asal.rollout import rollout_simulation

def calc_similarity_score(z, z_desc):
    """
    Calculates the supervisted target score from ASAL.
    The returned score should be minimized, since we add a minus sign here.
    """
    T, T2 = z.shape[0], z_desc.shape[0]
    assert T % T2 == 0
    z_desc = repeat(
        z_desc, "T2 D -> (k T2) D", k=T // T2
    )  # repeat to match shape, creating even intervals for each prompt

    kernel = z_desc @ z.T  # T, T
    return -jnp.diag(kernel).mean()


parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument(
    "--save_dir", type=str, default="./data/demo", help="path to save results to"
)

group = parser.add_argument_group("substrate")
group.add_argument(
    "--substrate", type=str, default="lenia", help="name of the substrate"
)
group.add_argument(
    "--rollout_steps",
    type=int,
    default=None,
    help="number of rollout timesteps, leave None for the default of the substrate",
)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def load_best_params(save_dir):
    """Load the best parameters from the saved pickle file."""
    best_params_path = os.path.join(save_dir, "best.pkl")
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
    return best_params[0]  # Extract best parameters


def main(args):
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps
    best_params = load_best_params(args.save_dir)
    # Using The Best Found Parameters
    definition = 2048
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=None,
        rollout_steps=args.rollout_steps,
        time_sampling="video",
        img_size=definition,
        return_state=False,
    )

    # Run simulation with best parameters
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = split(rng)
    rollout_data = rollout_fn(rng, best_params)

    # Convert frames from float [0,1] to uint8 [0,255] for video and save
    rgb_frames = np.array(rollout_data["rgb"])
    video_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)
    video_path = os.path.join(args.save_dir, "hd_video.mp4")
    imageio.mimsave(video_path, video_frames, fps=30)
    print(f"Video saved at: {video_path}")


if __name__ == "__main__":
    main(parse_args())