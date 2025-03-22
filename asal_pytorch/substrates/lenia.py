import torch
import torch.nn.functional as F
from .lenia_impl import LeniaImpl, ConfigLenia
from typing import Optional

def inv_sigmoid(x):
    return torch.log(x) - torch.log1p(-x)

class Lenia:
    """
    Higher-level Lenia class in PyTorch, mirroring the JAX-based Lenia substrate.
    """

    def __init__(
        self,
        grid_size: int = 128,
        center_phenotype: bool = True,
        phenotype_size: int = 48,
        start_pattern: str = "5N7KKM",
        clip1: float = float("inf"),
        clip2: float = float("inf")
    ):
        """
        :param grid_size:       Size of the Lenia simulation grid.
        :param center_phenotype: Whether to center the phenotype each step.
        :param phenotype_size:   The size of the cropped phenotype to record.
        :param start_pattern:     Which initial pattern to load from patterns.
        :param clip1:            Clipping range for the dynamic genotype parameters.
        :param clip2:            Clipping range for the initial cell genotype parameters.
        """
        self.grid_size = grid_size
        self.center_phenotype = center_phenotype
        self.phenotype_size = phenotype_size
        self.config_lenia = ConfigLenia(pattern_id=start_pattern, world_size=grid_size)
        self.lenia = LeniaImpl(self.config_lenia)  # Your PyTorch-based LeniaImpl

        self.clip1, self.clip2 = clip1, clip2

        # Load pattern and initialize
        init_carry, init_genotype, other_asset = self.lenia.load_pattern(self.lenia.pattern)
        self.init_carry = init_carry
        self.init_genotype = init_genotype

        # Convert the initial genotype into logit space (inv_sigmoid)
        # so it can be easily perturbed and then sigmoided back.
        self.base_params = inv_sigmoid(self.init_genotype)

    def default_params(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Creates a default random parameter tensor around zero, with the same shape
        as self.base_params. This matches the JAX code's `random.normal(...) * 0.1`.
        """
        shape = self.base_params.shape
        # If a torch.Generator is provided, use it; else just use global RNG
        return 0.1 * torch.randn(shape, generator=rng)

    def init_state(self, params: torch.Tensor, rng: Optional[torch.Generator] = None) -> dict:
        """
        Prepares the initial simulation state using the given `params`.
        This function:
         1. Splits params into dynamic (kernel) and initial-cell parts,
         2. Sigmoids them (with optional clipping),
         3. Expresses them into the Lenia simulation (carry),
         4. Optionally runs one step to get a non-blank initial image (like JAX code).
        """
        # Example indices: in JAX code, you used first 45 as dynamics, rest as init cells.
        # You may need to adjust the index (45) to match your actual genotype dimensioning.
        dynamic_len = self.lenia.n_params  # e.g., p*k in your PyTorch code
        base_dyn, base_init = self.base_params[:dynamic_len], self.base_params[dynamic_len:]
        params_dyn, params_init = params[:dynamic_len], params[dynamic_len:]
        device = params_dyn.device
        base_dyn = base_dyn.to(device)
        base_init = base_init.to(device)

        # Clip and then sigmoid, just like JAX code
        params_dyn = torch.sigmoid(base_dyn + params_dyn.clamp(-self.clip1, self.clip1))
        params_init = torch.sigmoid(base_init + params_init.clamp(-self.clip2, self.clip2))
        full_params = torch.cat([params_dyn, params_init], dim=0)

        # Express genotype into the Lenia world
        carry = self.lenia.express_genotype(self.init_carry, full_params)

        # Build a 'state' dict and run one step so the initial image isn't blank
        state = dict(carry=carry, img=None)
        state = self.step_state(state, full_params, rng=rng)
        return state

    def step_state(self, state: dict, params: torch.Tensor, rng: Optional[torch.Generator] = None) -> dict:
        """
        Advances the simulation one step. Records a phenotype image
        (cropped to phenotype_size) in `state['img']`.
        """
        carry = state["carry"]
        # The 'unused' argument in JAX is irrelevant in PyTorch, so we pass None
        carry, accum = self.lenia.step(
            carry,
            unused=None,
            phenotype_size=self.phenotype_size,
            center_phenotype=self.center_phenotype,
            record_phenotype=True
        )
        # accum.phenotype is the cropped image
        state["carry"] = carry
        state["img"] = accum.phenotype  # May be None if record_phenotype=False
        return state

    def render_state(self, state: dict, params: torch.Tensor, img_size: Optional[int] = None) -> torch.Tensor:
        """
        Returns the current image from the state. Optionally resizes it
        to `img_size x img_size`.
        """
        img = state["img"]
        if img is None:
            # Possibly means no phenotype was recorded, or step_state never called
            return torch.zeros((self.phenotype_size, self.phenotype_size, 3))

        if img_size is not None and img_size != img.shape[0]:
            # Use nearest-neighbor up/down-sampling
            img = img.permute(2, 0, 1).unsqueeze(0)  # BCHW
            img = F.interpolate(img, size=(img_size, img_size), mode='nearest')
            img = img[0].permute(1, 2, 0)  # back to HWC

        return img