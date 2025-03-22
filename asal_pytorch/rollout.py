import torch
from typing import Optional, Union, Tuple

def rollout_simulation(
    rng: Optional[torch.Generator],
    params: torch.Tensor,
    s0: Optional[dict] = None,
    substrate=None,
    fm=None,
    rollout_steps: int = 256,
    time_sampling: Union[str, int, Tuple[int, bool]] = 'final',
    img_size: int = 224,
    return_state: bool = False
):
    """
    Roll out a simulation under a PyTorch-based Lenia substrate, returning
    final or intermediate frames, embeddings, and optional internal state.

    Parameters
    ----------
    rng : torch.Generator or None
        RNG seed for the rollout. If None, global RNG is used.
    params : torch.Tensor
        Genotype parameters for the substrate.
    s0 : dict or None
        If provided, use as initial simulation state. Otherwise, call substrate.init_state(rng, params).
    substrate : object
        The substrate object (PyTorch-based), providing:
          - init_state(rng, params) -> state
          - step_state(rng, state, params) -> new_state
          - render_state(state, params, img_size) -> torch.Tensor image
    fm : object or None
        A "foundation model" object with a method fm.embed_img(image)-> embedding vector.
        If None, no embeddings are computed.
    rollout_steps : int
        Number of timesteps to run the simulation.
    time_sampling : str or int or (int, bool)
        - 'final': returns only final state data.
        - 'video': returns data at each timestep (a list).
        - int (K): returns K equally spaced states across the entire rollout.
        - (K, chunk_ends): if chunk_ends=True, sampling is offset to the end. Matches your JAX logic.
    img_size : int
        Height/width for rendered images (e.g. 224 for CLIP).
    return_state : bool
        If True, include the simulation state in each returned step. If False, only return image/embedding.

    Returns
    -------
    A dictionary or list of dictionaries with keys:
      - 'rgb':  (H, W, 3) or list thereof
      - 'z':    embedding or list thereof (if fm is not None)
      - 'state': dict or None (depending on return_state)
    """
    # Setup initial state
    if s0 is None:
        s0 = substrate.init_state(rng, params)

    # If no foundation model, no embedding
    def embed_img_fn(img):
        return None if fm is None else fm.embed_img(img)

    #'final' -> run rollout_steps, return only final frame + embedding
    if time_sampling == 'final':
        state = s0
        for _ in range(rollout_steps):
            state = substrate.step_state(rng, state, params)
        # Render final frame
        img = substrate.render_state(state, params, img_size=img_size)
        z = embed_img_fn(img)
        return {
            'rgb': img,
            'z': z,
            'state': state if return_state else None
        }

    # 'video' -> store data at each timestep
    elif time_sampling == 'video':
        # We'll store a dict of data at each step (0..rollout_steps-1)
        # final state is not strictly included unless we want it.
        data_list = []
        state = s0
        for step_i in range(rollout_steps):
            # Render current state
            img = substrate.render_state(state, params, img_size=img_size)
            z = embed_img_fn(img)
            data_entry = {
                'rgb': img,
                'z': z,
                'state': state if return_state else None
            }
            data_list.append(data_entry)
            # Step to next state
            state = substrate.step_state(rng, state, params)

        return data_list

    # K or (K, chunk_ends) -> run rollout_steps, then sample K intervals or chunk_ends
    elif isinstance(time_sampling, int) or (
        isinstance(time_sampling, tuple)
        and len(time_sampling) == 2
        and isinstance(time_sampling[0], int)
        and isinstance(time_sampling[1], bool)
    ):
        if isinstance(time_sampling, int):
            K = time_sampling
            chunk_ends = False
        else:
            K, chunk_ends = time_sampling

        # We'll store the entire trajectory to sample from it
        states = []
        state = s0
        states.append(state)
        for _ in range(rollout_steps):
            state = substrate.step_state(rng, state, params)
            states.append(state)

        # states list length = rollout_steps+1 (initial + each step)

        # Compute sampling indices
        chunk_size = rollout_steps // K
        if chunk_ends:
            # start sampling from chunk_size-1 to rollout_steps
            # i.e. [chunk_size, 2*chunk_size, ..., rollout_steps]
            idx_sample = list(range(chunk_size, rollout_steps+1, chunk_size))
        else:
            # [0, chunk_size, 2*chunk_size, ..., rollout_steps]
            idx_sample = list(range(0, rollout_steps+1, chunk_size))
            # ensure we don't exceed final index
            if idx_sample[-1] > rollout_steps:
                idx_sample[-1] = rollout_steps

        # For each sampled index, render
        data_list = []
        for idx in idx_sample:
            st = states[idx]
            img = substrate.render_state(st, params, img_size=img_size)
            z = embed_img_fn(img)
            data_entry = {
                'rgb': img,
                'z': z,
                'state': st if return_state else None
            }
            data_list.append(data_entry)

        return data_list

    else:
        raise ValueError(f"time_sampling {time_sampling} not recognized")