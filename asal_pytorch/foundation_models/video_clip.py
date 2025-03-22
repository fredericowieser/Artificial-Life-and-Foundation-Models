
import torch

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from PIL import Image

from typing import Any, Union, List
from functools import partial
from dataclasses import dataclass
from collections import namedtuple


from transformers import (
    AutoProcessor,
    FlaxCLIPModel,
    VideoCLIPModel,
    VideoCLIPProcessor,
)

class VideoCLIP:
    def __init__(
        self,
        model_name: str = "m-bain/videoclip",
        device: str = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the VideoCLIP model and processor for embedding text, images, and videos.
        """


        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Enable faster GPU matrix multiplications
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        self.device = device

        # Load model + processor
        self.processor = VideoCLIPProcessor.from_pretrained(model_name)
        self.model = VideoCLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype)

        if torch.__version__ >= "2.0":
            self.model = torch.compile(self.model)

        self.model.to(self.device)

    @torch.no_grad()
    def embed_txt(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get a normalized text embedding from VideoCLIP.
        """
        if isinstance(text, str):
            text = [text]

        # Prepare text inputs for VideoCLIP
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Extract text embedding: (batch_size, hidden_dim)
        text_emb = outputs.text_embeds  # e.g., shape: [B, D]

        # Convert to NumPy, then L2-normalize
        # text_emb_np = text_emb.cpu().numpy()
        # text_emb_normed = text_emb_np / np.linalg.norm(text_emb_np, axis=-1, keepdims=True)

        # return jnp.array(text_emb_normed)
        text_emb_normed = text_emb / text_emb.norm(dim=-1, keepdim=True)

        return text_emb_normed

    @torch.no_grad()
    def embed_img(self, image: Image.Image) -> torch.Tensor:
        """
        Hacky single-frame 'video' embedding from VideoCLIP.
        Internally treats the single image as a 1-frame video.
        """
        # VideoCLIP expects 'videos=', so wrap this image in a list
        frames = [image.convert("RGB")]

        # Process
        inputs = self.processor(
            text=None,
            videos=frames,  # single-frame "video"
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Extract video embedding: (batch_size, hidden_dim)
        vid_emb = outputs.video_embeds  # shape: [1, D]

        # Convert to NumPy, L2-normalize
        # vid_emb_np = vid_emb.cpu().numpy()
        # vid_emb_normed = vid_emb_np / np.linalg.norm(vid_emb_np, axis=-1, keepdims=True)

        # return jnp.array(vid_emb_normed)

        vid_emb_normed = vid_emb / vid_emb.norm(dim=-1, keepdim=True)

        return vid_emb_normed

    @torch.no_grad()
    def embed_video(self, video_frames: List[np.ndarray], max_frames: int = 20) -> torch.Tensor:
        """
        Get a normalized video embedding by:
          - Converting frames (NumPy arrays) to PIL
          - Subsampling if too many frames
          - Passing them all at once to VideoCLIP
        """
        # # Convert each frame from NumPy to PIL
        # frames = [Image.fromarray(frame).convert("RGB") for frame in video_frames]
        frames = []
        for frame in video_frames:
            if isinstance(frame, Image.Image):
                frames.append(frame.convert("RGB"))
            else:
                # Assume frame is a NumPy array-like; convert via PIL
                frames.append(Image.fromarray(frame).convert("RGB"))

        # Subsample if too many frames
        if len(frames) > max_frames:
            step = max(1, len(frames) // max_frames)
            frames = frames[0::step][:max_frames]

        # Process as a single video
        inputs = self.processor(
            text=None,
            videos=frames,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Extract video embedding (batch_size, hidden_dim); typically shape [1, D]
        vid_emb = outputs.video_embeds

        # # Convert to NumPy, L2-normalize
        # vid_emb_np = vid_emb.cpu().numpy()
        # vid_emb_normed = vid_emb_np / np.linalg.norm(vid_emb_np, axis=-1, keepdims=True)

        # return jnp.array(vid_emb_normed)
        vid_emb_normed = vid_emb / vid_emb.norm(dim=-1, keepdim=True)

        return vid_emb_normed

