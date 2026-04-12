from typing import List, Optional
import torch
import torch.nn.functional as F

from openai import OpenAI
from app.config import get_provider_config, get_settings

from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class OpenAICompatibleEmbedder:

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        cfg = get_provider_config(settings, settings.embed_provider)
        self.client = OpenAI(api_key = cfg.api_key, base_url = cfg.base_url)
        self.model = model or settings.embed_model

    
    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model = self.model,
            input = text,
        )

        return resp.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        resp = self.client.embeddings.create(
            model = self.model,
            input = texts,
        )
        return [item.embedding for item in resp.data]
    

class CLIPMultimodalEmbedder:
    """
    CLIP 统一编码文本/图像，适合多模态检索。
    不直接处理视频；视频请先抽帧后调用 embed_images。
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device = Optional[str] | None = None) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text:str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0]


    def embed_texts(self, texts: str) -> List[List[float]]:
        if not texts:
            return []
        
        inputs = self.processor(
            text = texts,
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).to(self.device)

        with torch.inference_mode():
            feats = self.model.get_text_features(**inputs)
            feats = F.normalize(feats, p = 2, dim = -1)
        
        return feats.cpu().tolist()


    def embed_image(self, image_path: str) -> List[float]:
        vectors = self.embed_images([image_path])
        return vectors[0]

    
    def embed_image(self, image_paths: str) -> List[List[float]]:
        if not image_paths: 
            return []
        
        images = [Image.open(p).convert("RGB") for p in image_paths]

        try:
            inputs = self.processor(images = images, return_tensors = "pt").to(self.device)

            with torch.inference_mode():
                feats = self.model.get_image_features(**inputs)
                feats = F.normalize(feats, p = 2, dim = -1)

            return feats.cpu().tolist()
        
        finally:
            for img in images:
                img.close()
