import torch
import numpy as np
import jax.numpy as jnp
from typing import Union, List
from PIL import Image
from transformers import Blip2Processor, Blip2Model

class BLIP2:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = None,
        torch_dtype=torch.float16
    ):
        """
        Initialize the BLIP-2 model and processor for embedding text, images, and videos.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Enable faster GPU matrix multiplications
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch_dtype)


        if torch.__version__ >= "2.0":
            self.model = torch.compile(self.model)

        self.model.to(device)



    @torch.no_grad()
    def embed_txt(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get a normalized text embedding from BLIP-2.

        - We request hidden states from the language model
          and perform a simple mean-pooling over the sequence dimension.
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize text
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Tell BLIP-2 to return hidden states from the LM
        text_outputs = self.model.get_text_features(
            **inputs,
            output_hidden_states=True,   # crucial for grabbing embeddings
            return_dict=True
        )
        # text_outputs is a CausalLMOutputWithPast => it has .hidden_states but no .pooler_output

        # hidden_states is a tuple of shape (num_layers, batch_size, seq_len, hidden_dim)
        hidden_states = text_outputs.hidden_states
        last_hidden_state = hidden_states[-1]  # shape (batch_size, seq_len, hidden_dim)

        # Simple mean-pool across the sequence dimension
        text_emb = last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_dim)

        # Convert to NumPy, then L2-normalize with JAX
        # text_emb_np = text_emb.detach().cpu().numpy()
        # text_emb_normed = text_emb_np / np.linalg.norm(text_emb_np, axis=-1, keepdims=True)

        text_emb_normed = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb_normed

    @torch.no_grad()
    def embed_img(self, image: Image.Image) -> torch.Tensor:
        """
        Get a normalized image embedding from BLIP-2.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # BLIP-2's image encoder returns a BaseModelOutputWithPooling by default
        vision_outputs = self.model.get_image_features(**inputs)

        # We can use 'pooler_output' to get a single [batch_size, hidden_dim] embedding
        img_emb = vision_outputs.pooler_output  # shape: (batch_size, hidden_dim)

        # img_emb_np = img_emb.detach().cpu().numpy()
        # img_emb_normed = img_emb_np / np.linalg.norm(img_emb_np, axis=-1, keepdims=True)

        # return jnp.array(img_emb_normed)
        img_emb_normed = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb_normed

    @torch.no_grad()
    def embed_video(self, video_frames: List[np.ndarray], max_images: int = 20) -> torch.Tensor:
        """
        Get a normalized video embedding by:
          - Converting frames to PIL
          - Subsampling if too many frames
          - Averaging the frame embeddings from BLIP-2's image encoder
        """
        # Convert each frame from NumPy array to PIL
        # frames = [Image.fromarray(frame).convert("RGB") for frame in video_frames]
        frames = []
        for frame in video_frames:
            if isinstance(frame, Image.Image):
                frames.append(frame.convert("RGB"))
            else:
                # Assume frame is a NumPy array-like; convert via PIL
                frames.append(Image.fromarray(frame).convert("RGB"))

        # Optionally subsample to avoid passing too many frames
        if len(frames) > max_images:
            step = max(1, len(frames) // max_images)
            frames = frames[0::step][:max_images]

        # Process frames in a single batch
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features (BaseModelOutputWithPooling)
        vision_outputs = self.model.get_image_features(**inputs)
        # shape: (num_frames, hidden_dim)
        frame_embs = vision_outputs.pooler_output

        # Average across frames -> single vector per video
        vid_emb = frame_embs.mean(dim=0, keepdims=True)

        # Convert to NumPy, L2-normalize with JAX
        # vid_emb_np = vid_emb.detach().cpu().numpy()
        # vid_emb_normed = vid_emb_np / np.linalg.norm(vid_emb_np, axis=-1, keepdims=True)

        # return jnp.array(vid_emb_normed)
        vid_emb_normed = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
        return vid_emb_normed

    