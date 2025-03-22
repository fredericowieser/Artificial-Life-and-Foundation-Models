import torch
from .lenia import Lenia

def create_substrate(substrate_name):
    """
    Create the substrate given a substrate name.
    The substrate parameterizes the space of simulations to search over.
    It has the following methods attached to it:
        - substrate.default_params(rng) to sample random parameters
        - substrate.init_state(rng, params) to sample the state from the initial state distribution
        - substrate.step_state(rng, state, params) to step the state forward one timestep
    
    Possible substrate names:
        - 'boids': Boids
        - 'lenia': Lenia
        - 'plife': ParticleLife
        - 'plife_plus': ParticleLifePlus
        - 'plenia': ParticleLenia
        - 'dnca': DNCA
        - 'nca_d1': NCA with d_state=1
        - 'nca_d3': NCA with d_state=3
        - 'gol': GameOfLife
    """
    rollout_steps = 1000
    if substrate_name=='lenia':
        substrate = substrate = Lenia(
            grid_size=128,
            center_phenotype=True,
            phenotype_size=64,
            start_pattern="5N7KKM",
            clip1=1.0,
            clip2=1.0
        )
        rollout_steps = 256
    else:
        raise ValueError(f"Unknown substrate name: {substrate_name}")
    substrate.name = substrate_name
    substrate.rollout_steps = rollout_steps
    return substrate

class FlattenSubstrateParameters:
    """
    Flattens the parameters of a given Lenia substrate into a 1D tensor
    suitable for evolutionary optimization. Mirrors the functionality of
    the JAX-based FlattenSubstrateParameters, but in PyTorch.

    Usage:
      substrate = Lenia(...)               # Your PyTorch-based Lenia
      flat_substrate = FlattenSubstrateParameters(substrate)
      x0 = flat_substrate.default_params() # flattened genotype
      ...
    """

    def __init__(self, substrate):
        """
        :param substrate: An instance of your higher-level Lenia substrate
                          (the PyTorch version).
        """
        self.substrate = substrate
        # We create a default set of params once, to learn shape info.
        with torch.no_grad():
            default_p = self.substrate.default_params()
        # Store shape for flatten/unflatten
        self.original_shape = default_p.shape
        # Count total params
        self.n_params = default_p.numel()

    def flatten_single(self, params: torch.Tensor) -> torch.Tensor:
        """Flatten a single parameter tensor into shape (n_params,)."""
        return params.view(-1)

    def reshape_single(self, flat_params: torch.Tensor) -> torch.Tensor:
        """Reshape a flat parameter vector back to the original param shape."""
        return flat_params.view(self.original_shape)

    def default_params(self, rng: torch.Generator = None) -> torch.Tensor:
        """
        Returns a flat version of the default parameters.
        Equivalent to Flatten -> substrate.default_params().
        """
        params = self.substrate.default_params(rng)
        return self.flatten_single(params)

    def init_state(self, rng: torch.Generator, params: torch.Tensor):
        """
        Unflattens `params` and calls substrate.init_state(...).
        Returns the substrate's initial simulation state.
        """
        params = self.reshape_single(params)
        return self.substrate.init_state(params, rng)

    def step_state(self, rng: torch.Generator, state: dict, params: torch.Tensor):
        """
        Unflattens `params` and calls substrate.step_state(...).
        Returns the new state after one simulation step.
        """
        params = self.reshape_single(params)
        return self.substrate.step_state(state, params, rng)

    def render_state(self, state: dict, params: torch.Tensor, img_size=None):
        """
        Unflattens `params` and calls substrate.render_state(...).
        Returns a rendered image (phenotype).
        """
        params = self.reshape_single(params)
        return self.substrate.render_state(state, params, img_size)

    def __getattr__(self, name):
        """
        Fallback: delegate any undefined attribute lookups
        directly to the substrate object.
        """
        return getattr(self.substrate, name)