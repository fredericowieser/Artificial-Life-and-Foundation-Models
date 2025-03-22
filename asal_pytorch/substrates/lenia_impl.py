import torch
import torch.nn.functional as F
from dataclasses import dataclass
from .lenia_patterns import (
    patterns,
    Carry,
    Param,
    Asset,
    Temp,
    Stats,
    Accum,
    Others,
)

bell = lambda x, mean, stdev: torch.exp(-((x - mean) / stdev) ** 2 / 2)
growth = lambda x, mean, stdev: 2 * bell(x, mean, stdev) - 1

@dataclass
class ConfigLenia:
	# Init pattern
	pattern_id: str = "VT049W"

	# World
	world_size: int = 128
	world_scale: int = 1

	# Simulation
	n_step: int = 200

	# Genotype
	n_params_size: int = 3
	n_cells_size: int = 32

class LeniaImpl:
    def __init__(self, config):
        """
        PyTorch implementation of Lenia Initialization
        """
        self._config = config
        self.pattern = patterns[self._config.pattern_id]

        # Genotype
        self.n_kernel = len(self.pattern["kernels"])  # Number of kernels (k)
        self.n_channel = len(self.pattern["cells"])  # Number of channels (c)
        self.n_params = self._config.n_params_size * self.n_kernel  # Total genotype parameters
        self.n_cells = self._config.n_cells_size ** 2 * self.n_channel  # Total number of embryo cells
        self.n_gene = self.n_params + self.n_cells  # Total genotype size

    def create_world_from_cells(self, cells):
        """
        Create a world tensor from the given cell configuration.
        """
        mid = self._config.world_size // 2

        # Scale the cells
        scaled_cells = cells.repeat((self._config.world_scale, self._config.world_scale, 1))  # Repeat along spatial dims
        cy, cx = scaled_cells.shape[:2]

        # Create an empty world tensor
        A = torch.zeros((self._config.world_size, self._config.world_size, self.n_channel), dtype=torch.float32)

        # Place scaled cells in the center of the world
        A[mid - cx // 2: mid + cx - cx // 2, mid - cy // 2: mid + cy - cy // 2, :] = scaled_cells

        return A

    def load_pattern(self, pattern):
        """
        Load a pattern and preprocess its parameters for simulation.
        """
        # Unpack pattern data
        cells = torch.tensor(pattern['cells'], dtype=torch.float32).permute(1, 2, 0)  # (y, x, c)
        kernels = pattern['kernels']
        R = pattern['R'] * self._config.world_scale
        T = pattern['T']

        # Extract kernel parameters
        m = torch.tensor([k['m'] for k in kernels], dtype=torch.float32)  # (k,)
        s = torch.tensor([k['s'] for k in kernels], dtype=torch.float32)  # (k,)
        h = torch.tensor([k['h'] for k in kernels], dtype=torch.float32)  # (k,)
        init_params = torch.vstack([m, s, h])  # (p, k)

        # Generate reshaping arrays
        reshape_c_k = torch.zeros((self.n_channel, self.n_kernel))  # (c, k)
        reshape_k_c = torch.zeros((self.n_kernel, self.n_channel))  # (k, c)
        for i, k in enumerate(kernels):
            reshape_c_k[k['c0'], i] = 1.0
            reshape_k_c[i, k['c1']] = 1.0

        # Compute kernel-related matrices
        mid = self._config.world_size // 2
        x_range = torch.arange(-mid, mid, dtype=torch.float32) / R
        X = torch.meshgrid(x_range, x_range, indexing='ij')  # (y, x) coordinates
        X = torch.stack(X, dim=0)
        D = torch.sqrt(X[0]**2 + X[1]**2)  # Distance matrix
        Ds = [D * len(k['b']) / k['r'] for k in kernels]  # Scaled distances per kernel
        Ks = [(D < len(k['b'])) * torch.tensor(k['b'])[torch.clamp(D.long(), max=len(k['b'])-1)] * bell(D % 1, 0.5, 0.15) for D, k in zip(Ds, kernels)]
        K = torch.stack(Ks, dim=-1)  # (y, x, k)
        nK = K / K.sum(dim=(0, 1), keepdim=True)  # Normalize kernels
        fK = torch.fft.fft2(torch.fft.fftshift(nK, dim=(0, 1)), dim=(0, 1))  # FFT of kernels

        # Pad pattern cells into initial cells
        cy, cx = cells.shape[:2]
        py, px = self._config.n_cells_size - cy, self._config.n_cells_size - cx
        init_cells = F.pad(cells, (0, 0, px//2, px-px//2, py//2, py-py//2))  # (e, e, c)

        # Create world from initial cells
        A = self.create_world_from_cells(init_cells)

        # Pack initial data
        init_carry = Carry(
            world=A,
            param=Param(m, s, h),
            asset=Asset(fK, X, reshape_c_k, reshape_k_c, R, T),
            temp=Temp(torch.zeros(2), torch.zeros(2, dtype=torch.int), torch.zeros(2, dtype=torch.int), 0.0),
        )
        init_genotype = torch.cat([init_params.flatten(), init_cells.flatten()])
        other_asset = Others(D, K, cells, init_cells)
        return init_carry, init_genotype, other_asset

    def express_genotype(self, carry, genotype):
        """
        Express the genotype by reshaping parameters and generating the world state.
        """
        params = genotype[:self.n_params].reshape((self._config.n_params_size, self.n_kernel))
        cells = genotype[self.n_params:].reshape((self._config.n_cells_size, self._config.n_cells_size, self.n_channel))

        m, s, h = params
        A = self.create_world_from_cells(cells)

        carry = carry._replace(world=A)
        carry = carry._replace(param=Param(m, s, h))
        return carry

    def step(self, carry, unused, phenotype_size, center_phenotype, record_phenotype):
        """
        Optimized single Lenia step using PyTorch.
        """
        # Unpack data from last step
        A = carry.world
        m, s, h = carry.param
        fK, X, reshape_c_k, reshape_k_c, R, T = carry.asset
        last_center, last_shift, total_shift, last_angle = carry.temp

        # Ensure parameters are on the same device as A (assumes gradients aren’t required)
        device = A.device
        m = m.to(device)
        s = s.to(device)
        h = h.to(device)
        
        # Combine unsqueeze calls via view (assumes m, s, h are 1D tensors of length k)
        m = m.view(1, 1, -1)  # shape: (1, 1, k)
        s = s.view(1, 1, -1)
        h = h.view(1, 1, -1)

        # Precompute constants
        invT = 1.0 / T
        mid = self._config.world_size // 2
        half_size = phenotype_size // 2

        # Center world by rolling using negative last_shift (convert tensor directly)
        A = torch.roll(A, shifts=(-last_shift).tolist(), dims=(-3, -2))

        # FFT of the world state
        fA = torch.fft.fft2(A, dim=(-3, -2))
        
        # Convert reshape_c_k to complex only once if needed (ideally, precompute this outside the loop)
        if not torch.is_complex(reshape_c_k):
            reshape_c_k = reshape_c_k.to(torch.complex64)
        
        # Use torch.matmul instead of einsum: fA is (y, x, c) and reshape_c_k is (c, k)
        fA_k = torch.matmul(fA, reshape_c_k)  # shape: (y, x, k)
        
        # Inverse FFT to get convolution results; take the real part.
        U_k = torch.real(torch.fft.ifft2(fK * fA_k, dim=(-3, -2)))

        # Inline the growth function: growth(x) = 2*exp(-((x-m)/s)**2/2)-1
        tmp = (U_k - m) / s
        bell_val = torch.exp(-0.5 * (tmp ** 2))
        growth_val = 2 * bell_val - 1
        G_k = growth_val * h

        # Use torch.matmul instead of einsum: G_k (y, x, k) multiplied by reshape_k_c (k, c)
        G = torch.matmul(G_k, reshape_k_c)

        # Compute next world state with clamping
        next_A = torch.clamp(A + invT * G, 0, 1)

        # Calculate the center: weighted sum of positions (assumes X is shape (2, y, x))
        m00 = A.sum()
        sum2d = next_A.sum(dim=-1, keepdim=True)  # shape: (y, x, 1)
        # Rearranging dimensions to multiply with X (which is (2, y, x))
        center = (sum2d.squeeze(-1) * X).sum(dim=(-2, -1)) / m00
        shift = (center * R).to(torch.int)
        total_shift = total_shift + shift

        # Get phenotype if needed
        if record_phenotype:
            if center_phenotype:
                phenotype = next_A
            else:
                phenotype = torch.roll(next_A, shifts=(total_shift - shift).tolist(), dims=(0, 1))
            phenotype = phenotype[mid - half_size: mid + half_size, mid - half_size: mid + half_size]
        else:
            phenotype = None

        # Calculate mass and velocity
        mass = m00 / (R * R)
        actual_center = center + total_shift / R
        center_diff = center - last_center + last_shift / R
        linear_velocity = torch.linalg.norm(center_diff) * T

        # Calculate angular velocity
        angle = torch.atan2(center_diff[1], center_diff[0]) / torch.pi
        angle_diff = (angle - last_angle + 3) % 2 - 1
        # Use device-aware constant for the fallback
        angle_diff = torch.where(linear_velocity > 0.01, angle_diff, angle.new_zeros(()))
        angular_velocity = angle_diff * T

        # Check if world is empty or full using simpler slicing for borders
        is_empty = (next_A < 0.1).all(dim=(-3, -2)).any()
        borders = (next_A[0, :, :].sum() + next_A[-1, :, :].sum() +
                   next_A[:, 0, :].sum() + next_A[:, -1, :].sum())
        is_full = borders > 0.1
        is_spread = A[mid - half_size: mid + half_size, mid - half_size: mid + half_size].sum() / m00 < 0.9

        # Pack data for next step
        carry = carry._replace(world=next_A)
        carry = carry._replace(temp=Temp(center, shift, total_shift, angle))
 
        stats = Stats(mass, actual_center[1], -actual_center[0],
                    linear_velocity, angle, angular_velocity,
                    is_empty, is_full, is_spread)
        accum = Accum(phenotype, stats)
        return carry, accum
