import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor as AutoProcessor, CLIPModel

class CLIP:
    def __init__(self, device, clip_model="clip-vit-base-patch32"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = CLIPModel.from_pretrained(f"openai/{clip_model}").to(self.device)
        self.clip_model.eval()

        self.img_mean = torch.tensor(self.processor.image_processor.image_mean).view(3, 1, 1).to(self.device)
        self.img_std = torch.tensor(self.processor.image_processor.image_std).view(3, 1, 1).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def embed_img(self, img):
        """
        img: numpy array (H, W, C) with values in [0, 1]
        returns: tensor of shape (D)
        """
        if isinstance(img, Image.Image):
            img = self.transform(img).to(self.device)
        else:
            img = torch.tensor(img, device=self.device).permute(2, 0, 1)  # Convert HWC to CHW
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        img = (img - self.img_mean) / self.img_std
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            z_img = self.clip_model.get_image_features(pixel_values=img)
            z_img = F.normalize(z_img, p=2, dim=-1)

        return z_img.squeeze(0)

    def embed_txt(self, prompts):
        """
        prompts: list of strings
        returns: tensor of shape (B, D)
        """
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            z_text = F.normalize(z_text, p=2, dim=-1)

        return z_text