import torch

import jax.numpy as jnp
import numpy as np

import torchvision.transforms as T

from transformers import (
    AutoProcessor,
    FlaxCLIPModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from torchvision.transforms.functional import InterpolationMode


from PIL import Image

# Add your Hugging Face Token Here To Use Gemma3
from google.colab import userdata
hf_token = userdata.get('HF')

from typing import Any, Union, List

class Clip4Clip:
    def __init__(
        self,
        model_name: str = "Searchium-ai/clip4clip-webvid150k",
        device: str = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the Clip4Clip text & vision encoders and the tokenizer.
        """
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Text sub-model

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        if torch.__version__ >= "2.0":
            self.text_model = torch.compile(self.text_model)
        self.text_model.to(self.device)

        # Vision sub-model
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        # Compile model for speedup if using PyTorch 2.0+
        if torch.__version__ >= "2.0":
            self.vision_model = torch.compile(self.vision_model)
        self.vision_model.to(self.device)


        # TorchVision transforms for images/frames, as recommended in model card
        self.image_transform = T.Compose([
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    @torch.no_grad()
    def embed_txt(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (text_embeds is shape: [batch_size, hidden_dim])
        outputs = self.text_model(**inputs)
        text_emb = outputs["text_embeds"]  # or outputs[0]

        # L2-normalize (in torch, then convert to JAX)
        # text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        # text_emb_np = text_emb.cpu().numpy()
        # return jnp.array(text_emb_np)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb


    @torch.no_grad()
    def embed_img(self, image: Image.Image) -> torch.Tensor:
        # Apply transforms
        img_tensor = self.image_transform(image)
        # Add batch dimension => shape: [1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.vision_model(img_tensor)
        img_emb = outputs["image_embeds"]  # shape: (1, hidden_dim)

        # L2-normalize
        # img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        # img_emb_np = img_emb.cpu().numpy()
        # return jnp.array(img_emb_np)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb


    @torch.no_grad()
    def embed_video(self, video_frames: List[np.ndarray], max_frames: int = 20) -> torch.Tensor:
        # 1) Convert frames to PIL
        # pil_frames = [Image.fromarray(frm).convert("RGB") for frm in video_frames]
        pil_frames = []
        for frm in video_frames:
            if isinstance(frm, Image.Image):
                pil_frames.append(frm.convert("RGB"))
            else:
                pil_frames.append(Image.fromarray(frm).convert("RGB"))

        # 2) Subsample frames if necessary
        if len(pil_frames) > max_frames:
            step = max(1, len(pil_frames) // max_frames)
            pil_frames = pil_frames[0::step][:max_frames]

        # 3) Transform & embed each frame
        all_embs = []
        for frame in pil_frames:
            frame_tensor = self.image_transform(frame).unsqueeze(0).to(self.device)
            out = self.vision_model(frame_tensor)
            frame_emb = out["image_embeds"]  # shape: (1, hidden_dim)
            # L2-normalize each frame embedding in torch
            frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True)
            all_embs.append(frame_emb)

        if len(all_embs) == 0:
            # Edge case: if no frames, return zeros
            return jnp.zeros((1, self.vision_model.config.projection_dim))

        # 4) Stack + average => shape: (num_frames, hidden_dim) -> (1, hidden_dim)
        embs_stack = torch.cat(all_embs, dim=0)
        vid_emb = embs_stack.mean(dim=0, keepdim=True)  # shape: (1, hidden_dim)

        # 5) Final L2-normalize
        # vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
        # vid_emb_np = vid_emb.cpu().numpy()
        # return jnp.array(vid_emb_np)
        vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
        return vid_emb
