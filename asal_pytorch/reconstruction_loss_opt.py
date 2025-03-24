from functools import partial
import os
import torch
import numpy as np
import imageio
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger
from substrates import FlattenSubstrateParameters, Lenia
from foundation_models import create_foundation_model
from rollout import rollout_simulation
from asal_metrics import (
    calc_supervised_target_score,
    calc_supervised_target_softmax_score,
    calc_open_endedness_score,
    calc_reconstruction_loss
)
import cProfile
import pstats
import io
from tqdm import tqdm
from torch.func import vmap

from foundation_models.gemma3 import Gemma3Chat

FPS = 15

def asal(
    fm,
    device: torch.device,
    prompts: str = "a caterpillar",
    substrate=None,
    rollout_steps: int = 256,
    n_iters: int = 1000,
    save_dir: str = "./demo_run/",
    seed: int = 42,
    pop_size: int = 16,
    sigma: float = 0.1,
    coef_prompt: float = 0.0,
    coef_softmax: float = 0.0,
    coef_oe: float = 0.0,
    coef_rl: float = 0.0,
    bs: int = 1,
):
    """
    Performs a black-box optimization with CMA-ES over the flattened parameters
    of a Lenia substrate, guided by textual prompts via a foundation model `fm`.

    Parameters
    ----------
    fm : FoundationModel
        Foundation model with methods:
          - fm.embed_txt(list_of_strings) -> torch.Tensor shape (N_prompts, embed_dim)
          - fm.embed_img(image_tensor)    -> torch.Tensor shape (embed_dim,)
    prompts : str
        Semicolon-separated textual prompts (e.g. "a butterfly; an orange cat").
    substrate : FlattenSubstrateParameters
        PyTorch-based Lenia substrate, flattened for evolutionary search. If None, a default is created.
    rollout_steps : int
        Number of timesteps to simulate for each candidate solution.
    n_iters : int
        Number of iterations of CMA-ES to run.
    save_dir : str
        Folder path to store partial logs/checkpoints.
    seed : int
        Random seed for CMA-ES.
    pop_size : int
        Population size of CMA-ES.
    sigma : float
        Initial standard deviation (step size) for CMA-ES.
    coef_prompt : float
        Weight of the prompt-based term in the loss.
    coef_softmax : float
        Weight of the softmax-based term in the loss.
    coef_oe : float
        Weight of open-endedness term in the loss.
    coef_rl : float
        Weight of reconstruction term in the loss.
    bs : int
        Number of random seeds for each solution's rollout (batching). Example demonstrates single-seed.

    Returns
    -------
    dict
        A dictionary containing 'best_fitness', 'best_solution', and 'loss_log'.
    """
    if substrate is None:
        substrate = Lenia(
            grid_size=128,
            center_phenotype=True,
            phenotype_size=64,
            start_pattern="5N7KKM",
            clip1=1.0
        )
        substrate = FlattenSubstrateParameters(substrate)
    else:
        # Ensure substrate is an instance of FlattenSubstrateParameters
        if not hasattr(substrate, "n_params"):
            substrate = FlattenSubstrateParameters(substrate)

    # Convert textual prompts into embeddings
    prompt_list = prompts.split(";")
    z_txt = fm.embed_txt(prompt_list)  # shape: (N_prompts, embed_dim)

    # We define a partial rollout function to gather images+embeddings across time
    rollout_fn = partial(
        rollout_simulation,
        rng=None,          # We'll rely on inside calls or single-seed
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=rollout_steps,
        time_sampling="video",  # gather all frames
        img_size=224,
        return_state=False
    )
    # rollout_fn = torch.jit.script(rollout_fn)

    # Optimise rollout Function (Only Work on CUDA)
    if device.type == "cuda":
        batched_rollout_fn = vmap(rollout_simulation, in_dims=(0,))

    if coef_rl > 0.0:
        rollout_fn_rl = partial(
            rollout_simulation,
            s0=None,
            substrate=substrate,
            fm=None,
            rollout_steps=rollout_steps,
            time_sampling='video', # Capture all frames
            img_size=224,
            return_state=False
        )

        gemma = Gemma3Chat(
            model_id="google/gemma-3-4b-it",
            max_context_length=128000,
        )
        # batched_rollout_fn version as well ?

    # Create an EvoTorch Problem
    class LeniaProblem(Problem):
        def __init__(self, device):
            # We treat it as a minimization problem
            super().__init__(
                objective_sense="min",
                solution_length=substrate.n_params,
                # Optionally define initial_bounds. In JAX code, you had no explicit bounds,
                # so we can do None or some range. We'll do a small range for demonstration:
                initial_bounds=(-1.0, 1.0),
                # We can define device / dtype if needed:
                # device="cpu", dtype=torch.float32
                device=device
            )

        def _evaluate_batch(self, solutions: SolutionBatch) -> None:
            """
            Evaluate a batch of solutions. This method is called by EvoTorch each iteration.
            solutions.values has shape (batch_size, n_params).
            We must compute a float loss for each solution and call solutions.set_evals(...).
            """
            x = solutions.values  # shape: (batch_size, n_params)
            batch_size = x.shape[0]

            # We'll store the final loss in a 1D tensor of shape [batch_size].
            losses = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            if device.type == "cuda":
                rollout_data = batched_rollout_fn(x)
            else:
                for i in range(batch_size):
                    params_i = x[i]
                    rollout_data = rollout_fn(params=params_i)

            if coef_rl > 0.0:
                rng = torch.manual_seed(0)
                # use batched_rollout_fn version as well ???
                for i in range(batch_size):
                    params_i = x[i]
                    rollout_data_rl = rollout_fn_rl(rng, params=params_i)
                    # or try data = rollout_data(rng, params)
            
            for i in range(batch_size):
                # rollout_data is a list of dicts (time_sampling='video').
                # Each dict has: 'rgb' -> image, 'z' -> embedding
                # Collect the 'z' from each frame:
                z_frames = [f['z'] for f in rollout_data if f['z'] is not None]
                if len(z_frames) == 0:
                    # If fm.embed_img returned None, define z = None => no embeddings
                    z = None
                else:
                    # shape: (T, embed_dim)
                    z = torch.stack(z_frames, dim=0)

                # Compute the combined objective:
                # Weighted sum of different sub-losses
                loss_i = torch.tensor(0.0, device=x.device, dtype=x.dtype)

                if coef_prompt > 0.0:
                    loss_prompt = calc_supervised_target_score(z, z_txt)
                    loss_i += coef_prompt * loss_prompt

                if coef_softmax > 0.0:
                    loss_softmax = calc_supervised_target_softmax_score(z, z_txt)
                    loss_i += coef_softmax * loss_softmax

                if coef_oe > 0.0:
                    loss_oe = calc_open_endedness_score(z)
                    loss_i += coef_oe * loss_oe
                
                if coef_rl > 0.0:
                    rgb_frames = np.array(rollout_data_rl['rgb'])
                    video_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)
                    # imageio.mimsave(output_path, video_frames, fps=fps)
                    final_text = gemma.describe_video(
                        video_frames=video_frames,
                        max_images=20
                    )
                    z_desc = fm.embed_txt(final_text[:70])
                    loss_rl = calc_reconstruction_loss(z_txt, z_desc)
                    loss_i += coef_rl * loss_rl

                losses[i] = loss_i

            solutions.set_evals(losses)

    problem = LeniaProblem(device=device)

    # We'll use CMAES from evotorch.algorithms.
    # If you want to use diagonal CMA, pass `separable=True`.
    searcher = CMAES(
        problem,
        popsize=pop_size,
        stdev_init=sigma,
        #random_seed=seed,
    )

    # Attach a logger that prints to stdout
    logger = StdOutLogger(searcher)

    # Optionally create save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # We'll store the best solution each iteration in a Python list for demonstration
    best_losses = []

    for iteration in tqdm(range(n_iters)):
        # Evolve one iteration
        searcher.step()
        # Current best fitness in the population
        current_best = searcher.status["pop_best_eval"]

        best_losses.append(current_best)

        # Periodically save
        # e.g. every 10% or final iteration
        if iteration % max(1, (n_iters // 10)) == 0 or iteration == (n_iters - 1):
            print(f"[Iteration {iteration}] best_loss = {current_best:.4f}")

            # Save checkpoint
            # import pdb; pdb.set_trace()
            best_solution_vector = searcher.population[0].values.clone().detach().cpu()
            if save_dir:
                save_dir_iter = os.path.join(save_dir, f"iter_{iteration}")
                os.makedirs(save_dir_iter, exist_ok=True)
                ckpt_path = os.path.join(save_dir_iter, f"ckpt_iter_{iteration}.pt")
                torch.save({
                    "iter": iteration,
                    "best_solution_vector": best_solution_vector
                }, ckpt_path)
                torch.save({
                    "iter": iteration,
                    "best_fitness": current_best,
                    "best_solution": best_solution_vector,
                    "loss_log": np.array(best_losses),
                }, ckpt_path)
                # Save A Video Of The Current Best Solution
                frames_data = rollout_fn(params=best_solution_vector)
                # import ipdb; ipdb.set_trace()
                # Save the video
                # Convert the frames to NumPy arrays in [0..255]
                frames_float = [step_dict['rgb'].cpu().numpy() for step_dict in frames_data]
                frames_uint8 = [np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in frames_float]

                # Write to MP4 (H.264) or GIF
                # imageio handles the format by extension. This will write an mp4 video.
                video_path = os.path.join(save_dir_iter, f"video_{iteration+1}.mp4")
                with imageio.get_writer(video_path, fps=FPS, format='FFMPEG') as writer:
                    for frame in frames_uint8:
                        writer.append_data(frame)
                print(f"Saved Lenia simulation video to: {save_dir_iter}")

    # Summarize final results
    best_fitness = searcher.status["pop_best_eval"]
    best_solution = searcher.population[0].values.clone().detach().cpu()

    print("==== Optimization Finished ====")
    print(f"Best Fitness: {best_fitness:.5f}")
    print(f"Best Solution Param shape: {best_solution.shape}")

    return {
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "loss_log": best_losses
    }

if __name__=="__main__":
    def get_best_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    DEVICE = get_best_device()
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # Load a foundation model
    fm = create_foundation_model("clip", device=DEVICE)

    # Profile the ASAL function
    profiler = cProfile.Profile()
    profiler.enable()

    results = asal(
        fm=fm,
        device=DEVICE,
        prompts="a caterpillar",
        substrate=None,
        rollout_steps=256,
        n_iters=1,
        save_dir="./demo_run/",
        seed=42,
        pop_size=4,
        sigma=0.1,
        coef_prompt=1.0,
        coef_softmax=0.0,
        coef_oe=0.0,
        coef_rl=1.0,
        bs=1
    )

    profiler.disable()

    # Save and print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")  # Sort by cumulative time
    ps.print_stats(20)  # Show top 20 slowest functions
    print(s.getvalue())  # Print the profiling results

    print("\n==== ASAL Demo Finished ====")
    print("Best Fitness:", results["best_fitness"])
    print("Best Solution (first 10 dims):", results["best_solution"][:10])
    print("Loss Log:", results["loss_log"])