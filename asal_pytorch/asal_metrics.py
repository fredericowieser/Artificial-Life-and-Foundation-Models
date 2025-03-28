import torch
import torch.nn.functional as F


def calc_supervised_target_score(z: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    T, D = z.shape
    T2 = z_txt.shape[0]
    assert T % T2 == 0, f"Expected T multiple of T2, got T={T}, T2={T2}"

    # Repeat z_txt so that it matches shape T
    k = T // T2
    # shape -> (T, D)
    z_txt_repeated = z_txt.repeat(
        (k, 1)
    )  # or z_txt.unsqueeze(0).repeat(k,1,1).view(-1,D)

    # Compute kernel = z_txt_repeated @ z^T => shape (T, T)
    kernel = torch.matmul(z_txt_repeated, z.t())

    # Negative mean diagonal
    return -torch.diagonal(kernel, 0).mean()


def calc_reconstruction_loss(z_desc: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    T, D = z_desc.shape
    T2 = z_txt.shape[0]
    assert T % T2 == 0, f"Expected T multiple of T2, got T={T}, T2={T2}"

    # Repeat z_txt so that it matches shape T
    k = T // T2
    z_txt_repeated = z_txt.repeat((k, 1))  # shape -> (T, D)

    kernel = torch.matmul(z_txt_repeated, z_desc.t())  # (T, T)
    return -torch.diagonal(kernel, 0).mean()


def calc_supervised_target_softmax_score(
    z: torch.Tensor, z_txt: torch.Tensor, temperature_softmax: float = 0.01
) -> torch.Tensor:
    T, D = z.shape
    T2 = z_txt.shape[0]
    assert T % T2 == 0, f"Expected T multiple of T2, got T={T}, T2={T2}"

    # Repeat z_txt to match T
    k = T // T2
    z_txt_repeated = z_txt.repeat((k, 1))  # (T, D)

    # Dot-product kernel => shape (T, T)
    kernel = torch.matmul(z_txt_repeated, z.t())

    # Compute softmax across different dimensions
    # kernel / temperature_softmax => scaled logit
    loss_sm1 = F.softmax(kernel / temperature_softmax, dim=-1)  # shape (T, T)
    loss_sm2 = F.softmax(kernel / temperature_softmax, dim=-2)  # shape (T, T)

    # Negative log of diagonal entries
    # diag(loss_smX) has shape (T,)
    diag_sm1 = torch.diagonal(loss_sm1, 0)
    diag_sm2 = torch.diagonal(loss_sm2, 0)
    # Avoid log(0) => clamp
    loss_sm1 = -torch.log(diag_sm1.clamp_min(1e-12)).mean()
    loss_sm2 = -torch.log(diag_sm2.clamp_min(1e-12)).mean()

    return 0.5 * (loss_sm1 + loss_sm2)


def calc_open_endedness_score(z: torch.Tensor) -> torch.Tensor:
    # kernel shape: (T, T)
    kernel = torch.matmul(z, z.t())

    # Zero out the upper triangle including diagonal
    # so we only keep the strictly lower-triangular part
    # or use torch.tril(kernel, diagonal=-1)
    kernel_lower = torch.tril(kernel, diagonal=-1)

    # row-wise max => shape (T,)
    row_max, _ = kernel_lower.max(dim=-1)

    return row_max.mean()
