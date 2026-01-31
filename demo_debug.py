"""
Debug version of Self-Forcing demo without Flask.
"""

import os
import random
import time
import base64
import torch
from io import BytesIO
from PIL import Image
import numpy as np
from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.utils import generate_timestamp
from demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, DynamicSwapInstaller
from omegaconf import OmegaConf

# -------------------------
# Config / Model Initialization
# -------------------------
print("🚀 Starting initialization...")
low_memory = get_cuda_free_memory_gb(gpu) < 40
print(f"💾 Low memory mode: {low_memory} (Free VRAM: {get_cuda_free_memory_gb(gpu):.2f} GB)")

print("📄 Loading config files...")
config = OmegaConf.load("./configs/self_forcing_dmd.yaml")
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

print("📝 Initializing text encoder...")
text_encoder = WanTextEncoder()
text_encoder.eval()
text_encoder.to(dtype=torch.bfloat16)

print("🔧 Loading transformer model (this may take a while)...")
transformer = WanDiffusionWrapper(is_causal=True)
state_dict = torch.load('./checkpoints/self_forcing_dmd.pt', map_location="cpu")
print("✅ Transformer checkpoint loaded, applying weights...")
transformer.load_state_dict(state_dict['generator_ema'])
transformer.eval()
transformer.to(dtype=torch.float16)
print("✅ Transformer initialized")

# Initialize VAE
def initialize_vae_decoder():
    print("🎨 Loading VAE decoder (this may take a while)...")
    vae = VAEDecoderWrapper()
    vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
    print("✅ VAE checkpoint loaded, extracting decoder weights...")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value
    vae.load_state_dict(decoder_state_dict)
    vae.eval()
    vae.to(dtype=torch.float16)
    vae.to(gpu)
    vae.requires_grad_(False)
    print("✅ VAE decoder initialized")
    return vae

vae_decoder = initialize_vae_decoder()

print("🔗 Creating inference pipeline...")
pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)

print("🚚 Moving models to GPU...")
if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    print("✅ Text encoder installed with dynamic swap")
else:
    text_encoder.to(gpu)
    transformer.to(gpu)
    print("✅ Models moved to GPU")

print("✨ All models initialized successfully!")

# -------------------------
# Helper functions
# -------------------------
frame_number = 0
anim_name = "debug_video"
frame_rate = 6

def tensor_to_base64_frame(frame_tensor):
    """Save frame tensor as image and return path (no base64 needed for debug)."""
    global frame_number, anim_name
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))
    image = Image.fromarray(frame)
    output_dir = f"./images/{anim_name}"
    os.makedirs(output_dir, exist_ok=True)
    frame_number += 1
    path = f"{output_dir}/{anim_name}_{frame_number:03d}.jpg"
    image.save(path)
    return path

def generate_mp4_from_images(image_directory, output_video_path, fps=24):
    import subprocess
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, anim_name, anim_name + '_%03d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Video saved to {output_video_path}")

# -------------------------
# Main generation function
# -------------------------
@torch.no_grad()
def generate_video_stream(prompt="A test prompt", seed=None, enable_torch_compile=False, enable_fp8=False):
    print("generate_video_stream")
    global frame_number, anim_name, frame_rate
    if seed is None:
        seed = random.randint(0, 2**32)

    print(f"🎬 Starting generation for prompt: {prompt}")
    total_frames_sent = 0

    # Text encoding
    conditional_dict = text_encoder(text_prompts=[prompt])
    for key, value in conditional_dict.items():
        conditional_dict[key] = value.to(dtype=torch.float16)
    if low_memory:
        move_model_to_device_with_memory_preservation(text_encoder, target_device=gpu, preserved_memory_gb=get_cuda_free_memory_gb(gpu)+5)

    # Noise initialization
    rnd = torch.Generator(gpu).manual_seed(seed)
    noise = torch.randn([1, 21, 16, 60, 104], device=gpu, dtype=torch.float16, generator=rnd)

    # KV / Cross-attention cache
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)

    num_blocks = 7
    current_start_frame = 0
    all_num_frames = [pipeline.num_frame_per_block] * num_blocks 
    vae_cache = ZERO_VAE_CACHE
    for i in range(len(vae_cache)):
        vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.float16)

    for idx, current_num_frames in enumerate(all_num_frames):
        print(f"🔄 Processing block {idx+1}/{len(all_num_frames)}")
        noisy_input = noise[:, current_start_frame:current_start_frame+current_num_frames] # (b, 3, c, h, w)

        # Denoising loop
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones([1, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep # (1, 3)

            if index < len(pipeline.denoising_step_list) - 1: # not the last timestep
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input, # (b, 3, c, h, w)
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                ) # (b, 3, c, h, w)
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1), # (b * 3, c, h, w)
                    torch.randn_like(denoised_pred.flatten(0, 1)), # (b * 3, c, h, w)
                    next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )

        # Update KV cache for next block
        if idx != len(all_num_frames) - 1:
            transformer(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=torch.zeros_like(timestep),
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

        # Decode and save frames
        pixels, vae_cache = vae_decoder(denoised_pred.half(), *vae_cache)
        if idx == 0:
            pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        for frame_idx in range(pixels.shape[1]):
            frame_path = tensor_to_base64_frame(pixels[0, frame_idx])
            total_frames_sent += 1
            print(f"Saved frame {frame_idx+1} -> {frame_path}")

        current_start_frame += current_num_frames

    print(f"🎉 Generation finished, total frames: {total_frames_sent}")
    generate_mp4_from_images("./images", "./videos/"+anim_name+".mp4", frame_rate)

# -------------------------
# Run debug generation
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 Starting video generation...")
    print("="*60 + "\n")

    os.makedirs("./images", exist_ok=True)
    os.makedirs("./videos", exist_ok=True)

    # 你可以改 prompt 或 seed 测试
    generate_video_stream(prompt="A magical forest at sunset", seed=42)
