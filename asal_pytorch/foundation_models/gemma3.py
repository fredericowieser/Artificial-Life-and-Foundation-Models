
import torch
from transformers import (
    AutoProcessor,
    FlaxCLIPModel,
    Gemma3ForConditionalGeneration,
)
from tqdm.auto import tqdm
from PIL import Image
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from torchvision.transforms import ToPILImage
class Gemma3Chat:
    """
    A condensed Gemma 3 chat-based replacement for the old LLaVA code.
    """

    def __init__(
        self,
        model_id="google/gemma-3-4b-it",
        device=None,
        torch_dtype=None,
        max_context_length=128000,
    ):
        """
        :param model_id: Hugging Face ID of the Gemma 3 VLM model.
        :param device: Device string, e.g. "cuda" or "cpu". Auto-chosen if None.
        :param torch_dtype: Torch dtype (e.g., float16). Auto-chosen if None.
        :param max_context_length: Max token length for text + images. Defaults to 128k for Gemma 3 4B+.
        """
        # if device is None:
        #     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # self.device = torch.device(device)
        
        # Enable faster GPU matrix multiplications
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        self.max_context_length = max_context_length

        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            token=token,
        )
        if torch.__version__ >= "2.0":
            self.model = torch.compile(self.model)

        self.model.to(self.device)

    def describe_video(
            self,
            video_frames,
            max_images=20,
            extract_prompt="Describe the video.",
            max_tokens=65,
        ):
        """
        Generates a description for a list of raw video frames (NumPy arrays).
        Frames are sampled up to `max_images` and processed by the Gemma 3 model.
        """
        if isinstance(video_frames, Image.Image):
            frames = [Image.fromarray(f).convert("RGB") for f in video_frames]
        elif isinstance(video_frames, torch.Tensor):
        
            to_pil = ToPILImage()
            frames = []
            for frame in video_frames:
                frames.append(to_pil(frame))
    
        if len(frames) > max_images:
            step = max(1, len(frames) // max_images)
            frames = frames[0::step][:max_images]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": extract_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img} for img in frames]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)