import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor as AutoProcessor, CLIPModel

IS_MPS = torch.backends.mps.is_available()

class CLIP:
    def __init__(self, device=None, clip_model="clip-vit-base-patch32"):
        # Automatically select best device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self.device = device
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = CLIPModel.from_pretrained(f"openai/{clip_model}").to(self.device).eval()
        if torch.__version__ >= "2.0":
            self.clip_model = torch.compile(self.clip_model)
        self.img_mean = torch.tensor(self.processor.image_processor.image_mean, device=self.device).view(3, 1, 1)
        self.img_std = torch.tensor(self.processor.image_processor.image_std, device=self.device).view(3, 1, 1)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # Faster and better rescaling
            transforms.ToTensor(),
        ])

    def embed_img(self, img):
        """
        Accepts a PIL.Image, numpy array, or torch.Tensor.
        For torch.Tensors, supports both single images (H, W, C) and batches (B, H, W, C).
        Returns: if input is batched, a tensor of shape (B, D); if a single image, returns (D,).
        """
        # PIL Image: use transform and add batch dimension.
        if isinstance(img, Image.Image):
            img = self.transform(img).to(self.device, memory_format=torch.channels_last, non_blocking=True)
            img = img.unsqueeze(0)  # (1, C, H, W)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:  # (H, W, C)
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.ndim == 4:  # (B, H, W, C)
                img = img.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected tensor shape: {img.shape}")
            img = img.to(self.device, non_blocking=True)
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(
                    img, size=(224, 224),
                    mode='bilinear',
                    align_corners=False,
                    **({} if IS_MPS else {"antialias": True})
                )
        else:
            # Assume numpy array
            img = torch.as_tensor(img, dtype=torch.float32, device=self.device)
            if img.ndim == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.ndim == 4:
                img = img.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected array shape: {img.shape}")
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(
                    img, size=(224, 224),
                    mode='bilinear',
                    align_corners=False,
                    **({} if IS_MPS else {"antialias": True})
                )
        
        # Normalize images
        img = (img - self.img_mean) / self.img_std

        with torch.no_grad():
            z_img = self.clip_model.get_image_features(pixel_values=img)
            z_img = F.normalize(z_img, p=2, dim=-1)
        
        # If batch size is one, return a 1D tensor; otherwise return the batch.
        return z_img.squeeze(0) if z_img.shape[0] == 1 else z_img

    def embed_txt(self, prompts):
        """
        prompts: list of strings
        returns: tensor of shape (B, D)
        """
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device, non_blocking=True)

        with torch.no_grad():
            z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            z_text = F.normalize(z_text, p=2, dim=-1)

        return z_text
