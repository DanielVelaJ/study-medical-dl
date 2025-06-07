#!/usr/bin/env python3
"""
Image-batch encoder helper for biomedical CLIP models.
Handles loading models and encoding batches of images efficiently.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from open_clip import create_model_from_pretrained
import requests
from io import BytesIO


def load_biomedclip() -> Tuple[torch.nn.Module, callable]:
    """Load OPEN-PMC-18M quality model (BiomedCLIP)"""
    print("🔬 Loading BiomedCLIP (OPEN-PMC-18M)...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.eval()
    return model, preprocess


def simulate_pmc6m_model(base_model: torch.nn.Module, noise_level: float = 0.05) -> torch.nn.Module:
    """
    Simulate PMC-6M baseline by adding controlled noise to represent quality difference.
    In practice, you'd load the actual PMC-CLIP model from WeixiongLin/PMC-CLIP
    """
    print(f"📊 Simulating PMC-6M baseline (noise_level={noise_level})...")
    
    # Create a copy of the model with added noise to vision encoder
    model_copy = torch.nn.Module()
    model_copy.encode_image = base_model.encode_image
    model_copy.original_encode_image = base_model.encode_image
    
    def noisy_encode_image(images):
        with torch.no_grad():
            embeddings = model_copy.original_encode_image(images)
            # Add controlled noise to simulate quality degradation
            noise = torch.randn_like(embeddings) * noise_level
            return embeddings + noise
    
    model_copy.encode_image = noisy_encode_image
    return model_copy


def encode_image_batch(
    model: torch.nn.Module, 
    preprocess: callable, 
    images: List[Image.Image],
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    Encode a batch of images using the given model.
    
    Args:
        model: The CLIP model to use for encoding
        preprocess: Preprocessing function for images
        images: List of PIL Images
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Numpy array of embeddings (n_images, embedding_dim)
    """
    model = model.to(device)
    embeddings = []
    
    print(f"⚡ Encoding {len(images)} images in batches of {batch_size}...")
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Preprocess batch
        batch_tensors = []
        for img in batch_images:
            try:
                tensor = preprocess(img).unsqueeze(0)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Warning: Failed to preprocess image {i}, using dummy tensor")
                # Create dummy tensor if preprocessing fails
                batch_tensors.append(torch.zeros(1, 3, 224, 224))
        
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        
        # Encode batch
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def load_sample_images(n: int = 200) -> List[Image.Image]:
    """
    Load sample biomedical images (simulating ROCO test set).
    
    Args:
        n: Number of images to load
        
    Returns:
        List of PIL Images
    """
    print(f"🖼️  Loading {n} sample biomedical images...")
    
    # Sample biomedical image URLs
    sample_urls = [
        "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/chest_X-ray.jpg",
        # Add more URLs or use a different strategy for loading images
    ]
    
    images = []
    for i in range(min(n, len(sample_urls))):
        try:
            response = requests.get(sample_urls[i % len(sample_urls)], timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image from URL, creating dummy image")
            # Create dummy image if loading fails
            dummy_img = Image.new('RGB', (224, 224), color='gray')
            images.append(dummy_img)
    
    # Fill remaining slots with variations of loaded images
    while len(images) < n:
        base_img = images[len(images) % len(sample_urls)]
        # Create slight variations (rotate, resize, etc.)
        if len(images) % 4 == 1:
            img = base_img.rotate(90)
        elif len(images) % 4 == 2:
            img = base_img.resize((200, 200)).resize((224, 224))
        elif len(images) % 4 == 3:
            img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = base_img
        images.append(img)
    
    return images[:n] 