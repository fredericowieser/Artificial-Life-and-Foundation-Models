import os

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torchvision.transforms import ToPILImage


def clean_gemma_output(raw_text):
    """
    Removes lines before "model", trims extra whitespace, and returns the rest.
    """
    # Split on newlines
    lines = raw_text.splitlines()
    cleaned_lines = []

    # Look for "model" line and record everything after
    start_collecting = False
    for line in lines:
        if "model" in line.strip().lower():
            start_collecting = True
            continue
        if start_collecting:
            # Skip empty lines or lines with only whitespace
            if line.strip():
                cleaned_lines.append(line.strip())

    # Combine everything back into a single string
    return " ".join(cleaned_lines)


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
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)

        # Enable faster GPU matrix multiplications
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using {torch_dtype} for Gemma 3 model.")
        self.torch_dtype = torch_dtype

        self.max_context_length = max_context_length
        
        config = AutoConfig.from_pretrained(model_id)
        config._attn_implementation = "eager"  # or "sdpa" to re-enable
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            config=config,
        )
        if torch.__version__ >= "2.0":
            self.model = torch.compile(self.model)

        self.model.to(self.device)

    def describe_video(
        self,
        video_frames,
        max_images=10,
        extract_prompt="Describe the video.",
        max_tokens=65,
        temperature=0.5,
    ):
        """
        Generates a description for a list of raw video frames (NumPy arrays).
        Frames are sampled up to `max_images` and processed by the Gemma 3 model.
        """
        if isinstance(video_frames, torch.Tensor):
            to_pil = ToPILImage()
            frames = []
            for frame in video_frames:
                frames.append(to_pil(frame))
        else:
            frames = [Image.fromarray(f).convert("RGB") for f in video_frames]

        if len(frames) > max_images:
            step = max(1, len(frames) // max_images)
            frames = frames[0::step][:max_images]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": extract_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device=self.device)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=None,
                    top_k=None,
                )
        return clean_gemma_output(
            self.processor.decode(output_ids[0], skip_special_tokens=True)
        )

    def generate_completion(
        self,
        instruction_prompt="",
        max_tokens=20,
        temperature=0.5,
    ):
        """
        Generates completion for input prompt.
        """

        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction_prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device=self.device)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=None,
                    top_k=None,
                )
        return clean_gemma_output(
            self.processor.decode(output_ids[0], skip_special_tokens=True)
        )

    def compare_videos(
        self,
        video_frames_1,
        video_frames_2,
        max_images=10,
        extract_prompt="Describe the video.",
        max_tokens=65,
        temperature=0.5,
    ):
        """
        Generates a comparison between two lists of raw video frames (NumPy arrays).
        Frames are sampled up to `max_images` and processed by the Gemma 3 model.
        """
        video_frames = [video_frames_1, video_frames_2]
        if isinstance(video_frames, torch.Tensor):
            to_pil = ToPILImage()
            frames = []
            for frame in video_frames:
                frames.append(to_pil(frame))
        else:
            frames = [Image.fromarray(f).convert("RGB") for f in video_frames]

        if len(frames) > max_images:
            step = max(1, len(frames) // max_images)
            frames = frames[0::step][:max_images]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": extract_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device=self.device)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0.0 else False,
                    top_p=None,
                    top_k=None,
                )
        return clean_gemma_output(
            self.processor.decode(output_ids[0], skip_special_tokens=True)
        )