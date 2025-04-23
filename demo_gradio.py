import asyncio
import platform

# Check if running on Windows BEFORE importing gradio or starting loops
if platform.system() == "Windows":
    print("Setting asyncio event loop policy to SelectorEventLoopPolicy for Windows.")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from diffusers_helper.hf_login import login

import os
import json
import time
import logging # Add logging import
# <<< ADDED IMPORTS >>>
import cv2
from tqdm import tqdm
# <<< END ADDED IMPORTS >>>

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import glob
import random
import io

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run, FIFOQueue
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
# <<< ADDED SCIKIT-IMAGE IMPORT (if needed for histogram matching later, keep it for now) >>>
from skimage import exposure # For potential future use / testing
# <<< END ADDED IMPORT >>>

# --- Add TaskManager Class and Instance --- START
import uuid
from threading import Thread, Lock
import copy

class TaskManager:
    def __init__(self):
        self.jobs = {}  # job_id -> job_info
        self.lock = Lock()
        self.input_queues = {} # job_id -> FIFOQueue for cancellation

    def _generate_job_id(self):
        return str(uuid.uuid4())

    def start_new_job(self, batch_params):
        job_id = self._generate_job_id()
        input_queue = FIFOQueue() # Create a dedicated queue for this job

        initial_state = {
            'status': 'Queued',
            'progress_percentage': 0,
            'current_image_path': None,
            'current_prompt': None,
            'current_preview_image': None, # Store preview image data if needed (base64?) or path
            'last_video_path': None,
            'error_message': None,
            'is_running': False,
            'is_finished': False,
            'is_cancelled': False,
            'job_params': batch_params # Store original params for reference
        }

        with self.lock:
            self.jobs[job_id] = initial_state
            self.input_queues[job_id] = input_queue

        # --- Setup Logging --- #
        log_file_path = os.path.join(outputs_folder, f"batch_run_{job_id}.log")
        initial_state['log_file_path'] = log_file_path # Store log path in state
        # ------------------- #

        # Prepare arguments for batch_worker
        # Pass log_file_path to batch_worker
        worker_args = (
            self, job_id, input_queue, log_file_path, # Pass manager, id, queue, and log path
            batch_params['input_dir'], batch_params['order'], batch_params['prompt_source'],
            batch_params['base_instruction'], batch_params['api_key'], batch_params['batch_n_prompt'],
            batch_params['batch_seed'], batch_params['batch_total_second_length'], batch_params['latent_window_size'],
            batch_params['batch_steps'], batch_params['cfg'], batch_params['gs'], batch_params['rs'],
            batch_params['batch_gpu_memory_preservation'], batch_params['batch_use_teacache'],
            batch_params['batch_mp4_crf'], batch_params['batch_schedule_type'], batch_params['batch_skip_last_n_sections'],
            # --- ADDED: Pass the correction flag from params --- #
            batch_params['apply_reinhard_batch'],
            # <<< ADDED: Pass in-loop correction flag >>>
            batch_params['correct_in_loop']
            # <<< END ADDED >>>
            # -------------------------------------------------- #
        )

        # Start the worker thread
        # Note: Corrected target function name here
        thread = Thread(target=batch_worker, args=worker_args, daemon=True)
        thread.start()

        print(f"[TaskManager] Started Job {job_id}")
        return job_id

    def start_single_job(self, single_params):
        """Starts a single image generation job using the worker function."""
        job_id = self._generate_job_id()
        input_queue = FIFOQueue() # Dedicated queue for cancellation

        # Basic initial state, similar to batch but simpler
        initial_state = {
            'status': 'Queued',
            'progress_percentage': 0,
            'last_video_path': None,
            'error_message': None,
            'is_running': False,
            'is_finished': False,
            'is_cancelled': False,
            'job_params': single_params # Store original params
        }

        with self.lock:
            self.jobs[job_id] = initial_state
            self.input_queues[job_id] = input_queue

        # --- Setup Logging for Single Job --- #
        # Reuse logger configuration? Or create a simpler one?
        # For now, use a standard logger name. Consider file logging later if needed.
        logger = logging.getLogger(f"single_job_{job_id}")
        # Basic config if no handlers exist (avoids duplicate logs on re-runs)
        if not logger.hasHandlers():
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             logger.addHandler(handler)
             logger.setLevel(logging.INFO)
        # ---------------------------------- #

        # Prepare arguments for the 'worker' function
        # Ensure the keys in single_params match the expected arguments of worker
        worker_args = (
            self, job_id, logger, # Pass TaskManager instance, job_id, logger
            job_id, # <<< ADDED: Pass job_id as main_job_id for single tasks
            # --- Single Image Params --- #
            single_params['input_image_np'],
            single_params['last_frame_image_np'],
            single_params['prompt'],
            single_params['n_prompt'],
            single_params['seed'],
            single_params['total_second_length'],
            single_params['latent_window_size'],
            single_params['steps'],
            single_params['cfg'], single_params['gs'], single_params['rs'],
            single_params['gpu_memory_preservation'],
            single_params['use_teacache'],
            single_params['mp4_crf'],
            single_params['schedule_type'],
            single_params['skip_last_n_sections'],
            single_params['injection_strength'], # Kept
            # <<< ADDED: Pass in-loop flag >>>
            single_params['correct_in_loop'],
            # <<< END ADDED >>>
            # --- Queues & Prefixes --- #
            input_queue, # Pass the dedicated input queue for cancellation
            # output_queue is removed from worker
            f"single_{job_id[:8]}_", # job_id_prefix
            "" # progress_desc_prefix (can be empty or customized)
        )

        # Start the worker thread targeting the 'worker' function
        thread = Thread(target=worker, args=worker_args, daemon=True)
        thread.start()

        print(f"[TaskManager] Started Single Job {job_id}")
        return job_id

    def update_job_state(self, job_id, updates):
        with self.lock:
            if job_id in self.jobs:
                # Ensure we don't overwrite critical final states
                if self.jobs[job_id].get('is_finished') or self.jobs[job_id].get('is_cancelled'):
                    if 'is_running' in updates and updates['is_running']:
                        # Don't allow setting running=True if already finished/cancelled
                        print(f"[TaskManager] Warning: Attempted to update finished/cancelled job {job_id} as running.")
                        updates.pop('is_running', None)

                self.jobs[job_id].update(updates)
                # Add derived states
                self.jobs[job_id]['is_running'] = updates.get('is_running', self.jobs[job_id].get('is_running', False))
                self.jobs[job_id]['is_finished'] = updates.get('is_finished', self.jobs[job_id].get('is_finished', False))
                self.jobs[job_id]['is_cancelled'] = updates.get('is_cancelled', self.jobs[job_id].get('is_cancelled', False))

                # If finished, error, or cancelled, ensure running is false
                if self.jobs[job_id]['is_finished'] or self.jobs[job_id]['error_message'] or self.jobs[job_id]['is_cancelled']:
                    self.jobs[job_id]['is_running'] = False

                # print(f"[TaskManager] Updated Job {job_id}: {updates}") # Verbose logging
            else:
                print(f"[TaskManager] Warning: Attempted to update non-existent job {job_id}")

    def get_job_state(self, job_id):
        with self.lock:
            # Return a copy to prevent modification outside the manager
            return copy.deepcopy(self.jobs.get(job_id))

    def request_stop_job(self, job_id):
        with self.lock:
            if job_id in self.input_queues:
                print(f"[TaskManager] Sending stop request to Job {job_id}")
                self.input_queues[job_id].push('end')
                # Optionally update state immediately to 'Cancelling...'
                if job_id in self.jobs and not self.jobs[job_id]['is_finished']:
                     self.jobs[job_id]['status'] = 'Cancellation requested...'
                     self.jobs[job_id]['is_running'] = False # Tentatively set running to false
            else:
                 print(f"[TaskManager] Warning: Attempted to stop non-existent job {job_id}")

    def request_skip_current_task(self, job_id):
        with self.lock:
            if job_id in self.input_queues:
                print(f"[TaskManager] Sending skip request to Job {job_id}")
                # Check if already skipping/cancelling?
                if job_id in self.jobs and self.jobs[job_id]['is_running']:
                    self.input_queues[job_id].push('skip')
                    self.jobs[job_id]['status'] = 'Skip requested for current task...'
                else:
                    print(f"[TaskManager] Job {job_id} is not running, cannot skip.")
            else:
                print(f"[TaskManager] Warning: Attempted to skip non-existent job {job_id}")

# Instantiate the TaskManager globally
task_manager = TaskManager()
# --- Add TaskManager Class and Instance --- END

# --- Add Reinhard Color Transfer Function --- START
def reinhard_color_transfer(source_bgr, target_bgr, clip=True, preserve_paper=True):
    """
    Applies color statistics (mean, std dev) from target to source image.
    Operates in LAB color space.

    Args:
        source_bgr: Source image (OpenCV BGR uint8 format).
        target_bgr: Target/Reference image (OpenCV BGR uint8 format).
        clip (bool): Whether to clip results to valid [0, 255] range.
        preserve_paper (bool): Handle potential zero divisions/multiplications.

    Returns:
        Transformed image (OpenCV BGR uint8 format).
    """
    # Convert to LAB
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype("float32")

    # Calculate statistics
    mean_src, std_dev_src = cv2.meanStdDev(source_lab)
    mean_tar, std_dev_tar = cv2.meanStdDev(target_lab)

    # Split source channels
    (l_src, a_src, b_src) = cv2.split(source_lab)

    # Subtract source means
    l_src -= mean_src[0][0]
    a_src -= mean_src[1][0] # Index 1 for a
    b_src -= mean_src[2][0] # Index 2 for b

    # Scale by standard deviations
    if preserve_paper:
        # Handle potential zero std devs
        std_dev_src[0][0] = 1.0 if std_dev_src[0][0] == 0 else std_dev_src[0][0]
        std_dev_src[1][0] = 1.0 if std_dev_src[1][0] == 0 else std_dev_src[1][0]
        std_dev_src[2][0] = 1.0 if std_dev_src[2][0] == 0 else std_dev_src[2][0]

    l_src = (std_dev_tar[0][0] / std_dev_src[0][0]) * l_src
    a_src = (std_dev_tar[1][0] / std_dev_src[1][0]) * a_src
    b_src = (std_dev_tar[2][0] / std_dev_src[2][0]) * b_src

    # Add target means
    l_src += mean_tar[0][0]
    a_src += mean_tar[1][0]
    b_src += mean_tar[2][0]

    # Merge channels
    transfer_lab = cv2.merge([l_src, a_src, b_src])

    # Clip to valid range
    if clip:
        transfer_lab = np.clip(transfer_lab, 0, 255)

    # Convert back to BGR
    transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer_bgr
# --- Add Reinhard Color Transfer Function --- END

# --- Add Tensor <-> OpenCV Conversion Helpers --- START
def tensor_to_cv2_list(tensor_bcthw):
    """Converts BCTHW [-1, 1] Tensor to a list of HWC BGR [0, 255] uint8 NumPy arrays."""
    if tensor_bcthw is None or tensor_bcthw.ndim != 5:
        return []
    # Assume B=1, remove batch dimension
    tensor_cthw = tensor_bcthw.squeeze(0)
    # Permute CTHW to THWC
    tensor_thwc = tensor_cthw.permute(1, 2, 3, 0)
    # Clamp and shift to [0, 2]
    tensor_thwc = torch.clamp(tensor_thwc, -1.0, 1.0) + 1.0
    # Scale to [0, 255]
    numpy_thwc = tensor_thwc.cpu().numpy() * 127.5
    # Convert to uint8
    numpy_thwc_uint8 = numpy_thwc.astype(np.uint8)
    # Split into list of HWC arrays and convert RGB to BGR
    frame_list = []
    for i in range(numpy_thwc_uint8.shape[0]): # Iterate through T dimension
        frame_hwc_rgb = numpy_thwc_uint8[i]
        frame_hwc_bgr = cv2.cvtColor(frame_hwc_rgb, cv2.COLOR_RGB2BGR)
        frame_list.append(frame_hwc_bgr)
    return frame_list

def cv2_list_to_tensor(list_of_bgr_arrays, device):
    """Converts a list of HWC BGR [0, 255] uint8 NumPy arrays to BCTHW [-1, 1] Tensor."""
    if not list_of_bgr_arrays:
        return None
    tensor_list = []
    for frame_bgr in list_of_bgr_arrays:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Convert to float tensor [0, 255]
        tensor_hwc = torch.from_numpy(frame_rgb).float().to(device)
        # Scale to [0, 2]
        tensor_hwc = tensor_hwc / 127.5
        # Shift to [-1, 1]
        tensor_hwc = tensor_hwc - 1.0
        # Permute HWC to CHW
        tensor_chw = tensor_hwc.permute(2, 0, 1)
        tensor_list.append(tensor_chw)
    # Stack along time dimension (T) -> CTHW
    tensor_cthw = torch.stack(tensor_list, dim=1)
    # Add batch dimension (B) -> BCTHW
    tensor_bcthw = tensor_cthw.unsqueeze(0)
    return tensor_bcthw
# --- Add Tensor <-> OpenCV Conversion Helpers --- END

# --- Add Core Correction Function --- START
def apply_reinhard_correction(original_video_path, reference_image_path, progress=gr.Progress()):
    """
    Applies Reinhard color correction to a video using a reference image.
    Saves the corrected video with a specific suffix.

    Args:
        original_video_path (str): Path to the input video file.
        reference_image_path (str): Path to the reference image file.
        progress (gr.Progress): Gradio progress tracker.

    Returns:
        str: Path to the corrected video file, or None if an error occurs.
    """
    if not original_video_path or not os.path.exists(original_video_path):
        print(f"[Color Correct Error] Original video path not found: {original_video_path}")
        return None
    if not reference_image_path or not os.path.exists(reference_image_path):
        print(f"[Color Correct Error] Reference image path not found: {reference_image_path}")
        return None

    try:
        # Load reference image
        ref_image = cv2.imread(reference_image_path)
        if ref_image is None:
            print(f"[Color Correct Error] Failed to load reference image: {reference_image_path}")
            return None

        # Open input video
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            print(f"[Color Correct Error] Failed to open video: {original_video_path}")
            return None

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[Color Correct Error] Video has no frames or failed to read count: {original_video_path}")
            cap.release()
            return None

        # Define output path
        output_filename_base = os.path.splitext(os.path.basename(original_video_path))[0]
        output_filename = f"{output_filename_base}_reinhard_corrected.mp4"
        output_path = os.path.join(os.path.dirname(original_video_path) or '.', output_filename)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"[Color Correct] Writing corrected video to: {output_path}")

        # Process frames with progress
        progress(0, desc="准备色彩校正...")
        # Wrap the range with tqdm for internal progress tracking if needed, Gradio handles UI
        for i in tqdm(range(total_frames), desc="色彩校正进度", unit="frame", disable=True): # Disable tqdm console output
            ret, frame = cap.read()
            if not ret:
                print(f"\n[Color Correct Warning] Reached end of video unexpectedly at frame {i}.")
                break

            # Apply correction
            corrected_frame = reinhard_color_transfer(frame, ref_image)

            # Write frame
            out.write(corrected_frame)

            # Update Gradio progress
            progress((i + 1) / total_frames, desc=f"色彩校正中 {i + 1}/{total_frames}")

        # Release resources
        cap.release()
        out.release()
        print(f"[Color Correct] Finished. Corrected video saved: {output_path}")
        # <<< MODIFIED: Return normalized absolute path >>>
        return os.path.abspath(os.path.normpath(output_path))
        # <<< END MODIFIED >>>

    except Exception as e:
        print(f"[Color Correct Error] Exception during color correction: {e}")

# --- Add Core Correction Function --- END

# --- Gemini API Placeholder and dependencies ---
# Requirements: pip install google-generativeai
# Remember to handle API keys securely (e.g., environment variables)
# Add error handling for API calls
try:
    import google.generativeai as genai
    # Configure API key once at the start if possible and desired
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if api_key:
    #     genai.configure(api_key=api_key)
    #     print("Gemini API Key configured from GOOGLE_API_KEY.")
    # else:
    #     print("Warning: GOOGLE_API_KEY environment variable not set. API calls might fail if key not provided in UI.")
    gemini_available = True
except ImportError:
    gemini_available = False
    print("Warning: google-generativeai library not found. Gemini API functionality will be disabled.")

# Function to call Gemini API (modified to accept API key)
def call_gemini_api(image_pil: Image.Image, base_instruction: str, api_key: str, logger,
                    # <<< ADDED: Optional last frame image >>>
                    last_image_pil: Image.Image | None = None
                    # <<< END ADDED >>>
                    ) -> str:
    """
    Function to call Gemini API.
    Uses the provided API key directly for this specific call.
    It's generally better practice to configure the API key once globally if possible,
    but this approach allows direct input from the UI.
    Uses the provided logger for output.
    Can now accept an optional second image (last_image_pil).
    """
    # <<< MODIFIED: Log based on whether last image is present >>>
    log_image_info = "first frame" if last_image_pil is None else "first and last frames"
    logger.info(f"[Gemini API] Received instruction: '{base_instruction}' for {log_image_info}.")
    # <<< END MODIFIED >>>

    if not gemini_available:
        logger.error("Error: google-generativeai library not installed.")
        return "Error: Gemini library not installed."

    if not api_key:
         logger.error("Error: No API Key provided for this call.")
         return f"Error: API Key is missing. Instruction was: '{base_instruction}'"

    try:
        # Configure the API key directly for this call
        genai.configure(api_key=api_key)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        try:
            image_pil.save(img_byte_arr, format='JPEG')
            mime_type = "image/jpeg"
        except OSError:
            img_byte_arr = io.BytesIO()
            image_pil.save(img_byte_arr, format='PNG')
            mime_type = "image/png"

        img_bytes = img_byte_arr.getvalue()

        # Prepare parts for multimodal input
        image_part = {
            "mime_type": mime_type,
            "data": img_bytes
        }
        text_part = {
            "text": base_instruction
        }

        # Select the appropriate model
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

        # Make the API call
        response = model.generate_content([text_part, image_part])

        # --- Robust check for blocked or empty response ---
        if not response.candidates:
            block_reason_msg = "Unknown reason (empty candidates list)."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason_msg = f"Reason: {response.prompt_feedback.block_reason}"
            error_msg = f"Error: Gemini API returned no candidates. Prompt possibly blocked. {block_reason_msg}. Instruction: '{base_instruction}'"
            logger.error(f"[Gemini API] {error_msg}")
            return error_msg

        candidate = response.candidates[0]

        if not candidate.content.parts:
            finish_reason_msg = "Unknown reason (empty parts list)."
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                 finish_reason_msg = f"Reason: {candidate.finish_reason.name}"
            safety_ratings_msg = ""
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                 ratings = [f"{r.category.name}={r.probability.name}" for r in candidate.safety_ratings if r.probability.name != 'NEGLIGIBLE']
                 if ratings:
                     safety_ratings_msg = f" SafetyRatings: [{', '.join(ratings)}]"

            error_msg = f"Error: Gemini API response candidate has empty parts. {finish_reason_msg}{safety_ratings_msg}. Instruction: '{base_instruction}'"
            logger.error(f"[Gemini API] {error_msg}")
            return error_msg
        # -----------------------------------------------------

        generated_prompt = candidate.content.parts[0].text.strip()

        if not generated_prompt:
             logger.warning("[Gemini API] Warning: Generated prompt text was empty after stripping.")
             return f"Error: Gemini API returned an empty prompt text. Instruction was: '{base_instruction}'"

        logger.info(f"[Gemini API] Generated prompt: '{generated_prompt}'")
        return generated_prompt

    except Exception as e:
        logger.exception(f"Error calling Gemini API: {e}")
        return f"Error generating prompt via API: {e}. Instruction was: '{base_instruction}'"

# --- End Gemini API Definition ---

# --- Settings Persistence --- A
SETTINGS_FILE = "framepack_settings.json"

def load_settings():
    """Loads settings from the JSON file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {SETTINGS_FILE}. Using default settings.")
            return {}
        except Exception as e:
            print(f"Warning: Error loading {SETTINGS_FILE}: {e}. Using default settings.")
            return {}
    else:
        return {}

def save_settings(new_settings, section):
    """Saves settings to the JSON file, updating a specific section."""
    current_settings = load_settings() # Load existing settings
    if section not in current_settings:
        current_settings[section] = {}
    current_settings[section].update(new_settings) # Update the section
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(current_settings, f, indent=4)
        # print(f"Settings saved to {SETTINGS_FILE}") # Optional: uncomment for debug
    except Exception as e:
        print(f"Error saving settings to {SETTINGS_FILE}: {e}")

# Load settings at the start
loaded_app_settings = load_settings()
single_settings = loaded_app_settings.get('single', {})
batch_settings = loaded_app_settings.get('batch', {})
# --- End Settings Persistence --- B

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(task_manager, job_id, logger, # Added logger
           main_job_id: str, # <<< ADDED main_job_id parameter
           input_image_np, last_frame_image_np, # Added last_frame_image_np
           prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs,
           gpu_memory_preservation, use_teacache, mp4_crf, schedule_type, skip_last_n_sections,
           injection_strength, # Kept for blending
           # <<< ADDED: Flag for in-loop correction >>>
           correct_in_loop: bool,
           # <<< END ADDED >>>
           input_queue, # Removed output_queue
           job_id_prefix='', progress_desc_prefix=''):

    # Helper to update state - now always uses task_manager
    def update_state(updates):
        # Simply call task_manager's update method
        # >>> CHANGED to use main_job_id <<<
        task_manager.update_job_state(main_job_id, updates)

    # Use the passed input_queue for cancellation/skip checks
    in_q = input_queue

    # Custom Exception for skipping
    class SkipTaskException(Exception):
        pass

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    worker_job_id = job_id # Use main job ID for consistency in logs

    logger.info(f"Worker starting task.") # W_LOG 1

    exception_occurred = False # Initialize exception flag
    output_filename = None # Initialize output_filename

    try:
        # --- Mark job as running --- #
        update_state({'status': 'Starting...', 'is_running': True})
        # ------------------------- #

        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        logger.info(f"Encoding text...") # W_LOG 2
        # update_state({'status': progress_desc_prefix + 'Text encoding...', 'progress_percentage': 0}) # Status updated via log

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        logger.info(f"Text encoded.") # W_LOG 3

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        # update_state({'status': progress_desc_prefix + 'Image processing...'}) # Status updated via log
        logger.info("Processing input image...")

        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        resized_input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        input_img_filename = os.path.join(outputs_folder, f'{job_id_prefix}{worker_job_id[:8]}_input.png')
        Image.fromarray(resized_input_image_np).save(input_img_filename)
        logger.info(f"Saved resized input image to {input_img_filename}")

        # <<< ADDED: Store reference image path in state >>>
        update_state({'reference_image_path': input_img_filename})
        # <<< END ADDED >>>

        # <<< ADDED: Load reference BGR image if correcting in loop >>>
        ref_image_bgr = None
        if correct_in_loop:
            try:
                ref_image_bgr = cv2.imread(input_img_filename)
                if ref_image_bgr is None:
                    logger.warning(f"[Worker Warn] Correct-in-loop enabled but failed to load reference image: {input_img_filename}. Correction disabled.")
                    correct_in_loop = False # Disable if load fails
                else:
                    logger.info(f"[Worker Info] Correct-in-loop enabled. Loaded reference image: {input_img_filename}")
            except Exception as ref_load_exc:
                logger.warning(f"[Worker Warn] Exception loading reference image {input_img_filename}: {ref_load_exc}. Correction disabled.")
                correct_in_loop = False # Disable on exception
        # <<< END ADDED >>>

        input_image_pt = torch.from_numpy(resized_input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        logger.info(f"Encoding VAE...") # W_LOG 4
        # update_state({'status': progress_desc_prefix + 'VAE encoding...'}) # Status updated via log

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        logger.info(f"VAE encoded.") # W_LOG 5

        # --- Encode Last Frame Image for VAE Latent (if provided) --- #
        end_latent = None
        # Only depends on whether the image was provided now
        if last_frame_image_np is not None:
            logger.info("Encoding last frame image for VAE latent (image provided)...")
            try:
                # Reuse the same resize/crop logic and dimensions as the first frame
                # (We already resized for CLIP Vision, but VAE needs tensor)
                resized_last_frame_np_for_vae = resize_and_center_crop(last_frame_image_np, target_width=width, target_height=height)
                last_image_pt = torch.from_numpy(resized_last_frame_np_for_vae).float() / 127.5 - 1
                last_image_pt = last_image_pt.permute(2, 0, 1)[None, :, None]

                if not high_vram:
                    # Ensure VAE is loaded if needed (might have been unloaded after start_latent)
                    load_model_as_complete(vae, target_device=gpu)

                end_latent = vae_encode(last_image_pt, vae)
                logger.info("Last frame VAE latent encoded.")
            except Exception as e:
                logger.error(f"Error encoding last frame VAE latent: {e}. Latent injection disabled.")
                end_latent = None
        # No need for elif here anymore
        # ---------------------------------------------------------- #

        # CLIP Vision
        logger.info(f"Encoding CLIP Vision...") # W_LOG 6
        # update_state({'status': progress_desc_prefix + 'CLIP Vision encoding...'}) # Status updated via log

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(resized_input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        logger.info(f"CLIP Vision encoded.") # W_LOG 7

        # --- Encode Last Frame Image (if provided) --- #
        last_frame_image_embedding = None
        if last_frame_image_np is not None:
            logger.info("Processing last frame image...")
            try:
                # Use the same height/width derived from the first frame
                resized_last_frame_np = resize_and_center_crop(last_frame_image_np, target_width=width, target_height=height)

                # Optional: Save the resized last frame for debugging
                # last_frame_filename = os.path.join(outputs_folder, f'{job_id_prefix}{worker_job_id[:8]}_last_frame.png')
                # Image.fromarray(resized_last_frame_np).save(last_frame_filename)
                # logger.info(f"Saved resized last frame image to {last_frame_filename}")

                if not high_vram:
                    # Ensure image encoder is loaded if needed
                    load_model_as_complete(image_encoder, target_device=gpu)

                last_frame_output = hf_clip_vision_encode(resized_last_frame_np, feature_extractor, image_encoder)
                last_frame_image_embedding = last_frame_output.last_hidden_state
                logger.info("Last frame image CLIP Vision encoded.")
            except Exception as e:
                logger.error(f"Error processing last frame image: {e}. Will use first frame embedding only.")
                last_frame_image_embedding = None # Fallback to None on error
        # -------------------------------------------- #

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        first_frame_image_embedding = image_encoder_last_hidden_state.to(transformer.dtype)
        if last_frame_image_embedding is not None:
            last_frame_image_embedding = last_frame_image_embedding.to(transformer.dtype)

        # Sampling
        # update_state({'status': progress_desc_prefix + 'Start sampling...'}) # Status updated via log
        logger.info("Starting sampling...")

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = None
        if schedule_type == SCHEDULE_LINEAR:
            latent_paddings = list(reversed(range(total_latent_sections)))
            logger.info(f"Using Linear Reversed Schedule: {latent_paddings}")
        elif schedule_type == SCHEDULE_DEFAULT:
            # Corrected Indentation Start
            if total_latent_sections > 4: # <<< INDENTED
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else: # <<< ALIGNED with if
                latent_paddings = list(reversed(range(total_latent_sections)))
            # Corrected Indentation End
            logger.info(f"Using Default Experience Schedule: {latent_paddings}")
        elif schedule_type == SCHEDULE_CONSTANT_ZERO: # New
            latent_paddings = [0] * total_latent_sections
            logger.info(f"Using Constant Zero Padding Schedule: {latent_paddings}")
        elif schedule_type == SCHEDULE_START_END_ZERO: # New
            if total_latent_sections >= 2:
                latent_paddings = [0] + [2] * (total_latent_sections - 2) + [0]
            else: # Fallback for 1 section
                latent_paddings = [0] * total_latent_sections
            logger.info(f"Using Start/End Zero Padding Schedule: {latent_paddings}")
        # <<< ADDED: Logic for the new anti-drift schedule >>>
        elif schedule_type == SCHEDULE_INV_LINEAR_ANTIDRIFT:
            latent_paddings = list(range(total_latent_sections))
            logger.info(f"Using Inverted Linear (Anti-Drift) Schedule: {latent_paddings}")
        # <<< END ADDED >>>
        else:
             latent_paddings = list(reversed(range(total_latent_sections))) # Default fallback remains Linear Reversed
             logger.warning(f"Unknown schedule type '{schedule_type}', defaulting to Linear Reversed Schedule: {latent_paddings}")

        # --- Section Skipping Logic --- # Now applies to all calculated schedules
        total_sections_calculated = len(latent_paddings)
        logger.info(f"Calculated {total_sections_calculated} sections with paddings: {latent_paddings}")
        # Re-insert the missing definition
        num_to_skip = int(skip_last_n_sections) if skip_last_n_sections is not None else 0

        if num_to_skip > 0 and total_sections_calculated > 0:
            if num_to_skip >= total_sections_calculated:
                logger.warning(f"Requested to skip {num_to_skip} sections, but only {total_sections_calculated} exist. Skipping all but the first section.")
                latent_paddings = latent_paddings[:1]
            else:
                 original_paddings = list(latent_paddings)
                 latent_paddings = latent_paddings[:-num_to_skip]
                 logger.info(f"Skipping last {num_to_skip} sections. Original paddings: {original_paddings}, Processing paddings: {latent_paddings}")

        if not latent_paddings:
             logger.error(f"Error: No latent sections remaining after skipping configuration. Cannot proceed.")
             return ('error', "No latent sections remaining after skipping configuration.")

        logger.info(f"Entering sampling loop (processing {len(latent_paddings)} sections).") # W_LOG 8
        total_processing_sections = len(latent_paddings)

        for idx, latent_padding in enumerate(latent_paddings):
            # --- Cancellation/Skip Check --- #
            signal = in_q.top()
            if signal == 'skip':
                in_q.pop()
                logger.warning(f"Skip signal received during generation.")
                raise SkipTaskException()
            elif signal == 'end':
                logger.warning(f"Cancel signal ('end') received during generation.")
                return ('cancel', None)
            # --------------------------- #

            is_last_section = (idx == len(latent_paddings) - 1)
            latent_padding_size = latent_padding * latent_window_size
            section_progress_prefix = f"[Section {idx + 1}/{total_processing_sections}] "

            logger.info(f"{section_progress_prefix}Sampling loop iteration: latent_padding={latent_padding}, size={latent_padding_size}, is_last={is_last_section}") # W_LOG 9

            # --- Capture output_filename within the loop --- #
            # Initialize here for scope if loop doesn't run or breaks early
            output_filename = None

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)

            # --- Latent Injection (Blended - if end_latent exists) --- #
            if idx == 0 and end_latent is not None:
                original_clean_latents_post = clean_latents_post.clone() # Store original

                # Clamp injection_strength to [0.0, 1.0]
                strength = max(0.0, min(float(injection_strength), 1.0))

                logger.info(f"{section_progress_prefix}Blending end_latent into clean_latents_post (Strength: {strength*100:.0f}%).")
                # Perform weighted average
                clean_latents_post = (original_clean_latents_post * (1.0 - strength) +
                                      end_latent.to(history_latents) * strength)
            # --------------------------------------------------------- #

            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # --- Select Image Embedding based on Switch Point (REVERSED LOGIC) --- #
            current_image_embedding = first_frame_image_embedding
            logger.info(f"{section_progress_prefix}Using First Frame Embedding for guidance.")

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                # --- Callback Cancellation/Skip Check --- #
                signal = in_q.top()
                if signal == 'skip':
                    in_q.pop()
                    logger.warning(f"Skip signal received via callback.")
                    raise SkipTaskException()
                elif signal == 'end':
                    logger.warning(f"Cancel signal ('end') received via callback.")
                    raise KeyboardInterrupt('Main batch cancelled')
                # ------------------------------------ #

                # Preview generation removed - focus on logging progress
                # preview = d['denoised']
                # ... preview processing ...

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'{progress_desc_prefix}Sampling {current_step}/{steps}'
                desc = f'{progress_desc_prefix}Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). Extending...'

                # Log progress instead of updating state directly
                # <<< REMOVED: Redundant logging causing interleaved output with tqdm bar >>>
                # if current_step % 5 == 0 or current_step == steps: # Log every 5 steps or on the last step
                #     logger.info(f"{hint} ({percentage}%). {desc}")

                # Update only percentage in state for coarse progress display if needed elsewhere
                update_state({'progress_percentage': percentage}) # Update state with percentage
                return

            # Wrap the sample_hunyuan call in its own try block
            try: # Correct indentation for try
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                        image_embeddings=current_image_embedding, # Use the selected embedding (always first frame now)
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
            # Corrected indentation for except blocks (now aligned with try)
            except SkipTaskException: # <<< ALIGNED with try
                # Re-raise to be caught by the outer try block
                raise # <<< INDENTED under except
            except KeyboardInterrupt: # <<< ALIGNED with try
                # Re-raise to be caught by the outer try block
                logger.warning(f"KeyboardInterrupt caught during sample_hunyuan, re-raising.") # <<< INDENTED under except
                raise
            except Exception as e: # <<< ALIGNED with try
                # Log and re-raise to be caught by the outer try block
                logger.exception(f"Error during sample_hunyuan call:") # W_LOG 10.1 # <<< INDENTED under except
                # Optionally wrap e in a custom exception if more context needed
                raise # Re-raise the original exception

            # --- This code runs only if sample_hunyuan succeeded --- #

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            # --- Another Cancellation/Skip Check before VAE decode --- #
            signal = in_q.top()
            if signal == 'skip':
                in_q.pop()
                logger.warning(f"{section_progress_prefix}Skip signal received before VAE decode.")
                raise SkipTaskException()
            elif signal == 'end':
                logger.warning(f"{section_progress_prefix}Cancel signal ('end') received before VAE decode.")
                return ('cancel', None)
            # --------------------------------------------------------- #

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            logger.info(f"{section_progress_prefix}Before VAE decode. History pixels is None: {history_pixels is None}") # W_LOG 11
            if history_pixels is None:
                # First section: Decode the initial latents
                decoded_pixels_tensor = vae_decode(generated_latents, vae).cpu() # Decode to CPU first

                # <<< ADDED: In-loop correction for first section >>>
                if correct_in_loop and ref_image_bgr is not None:
                    logger.info(f"  [Worker Correct-in-Loop] Applying Reinhard to first section ({decoded_pixels_tensor.shape[2]} frames)...")
                    start_corr_time = time.time()
                    decoded_cv2_list = tensor_to_cv2_list(decoded_pixels_tensor)
                    corrected_cv2_list = [reinhard_color_transfer(frame, ref_image_bgr) for frame in decoded_cv2_list]
                    history_pixels = cv2_list_to_tensor(corrected_cv2_list, device=cpu) # Keep corrected on CPU
                    logger.info(f"  [Worker Correct-in-Loop] First section correction took {time.time() - start_corr_time:.2f}s.")
                else:
                    history_pixels = decoded_pixels_tensor # Assign directly if no correction
                # <<< END ADDED >>>
            else:
                # Subsequent sections: Decode current section
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                logger.info(f"{section_progress_prefix}Decoding section latents slice [: {section_latent_frames}] from history {real_history_latents.shape} (Old Logic)...")
                current_pixels_tensor = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()

                # <<< ADDED: In-loop correction for subsequent sections >>>
                pixels_for_append = current_pixels_tensor # Default to original
                if correct_in_loop and ref_image_bgr is not None:
                    logger.info(f"  [Worker Correct-in-Loop] Applying Reinhard to current section ({current_pixels_tensor.shape[2]} frames) before append...")
                    start_corr_time = time.time()
                    current_cv2_list = tensor_to_cv2_list(current_pixels_tensor)
                    corrected_cv2_list = [reinhard_color_transfer(frame, ref_image_bgr) for frame in current_cv2_list]
                    pixels_for_append = cv2_list_to_tensor(corrected_cv2_list, device=cpu) # Keep corrected on CPU
                    logger.info(f"  [Worker Correct-in-Loop] Current section correction took {time.time() - start_corr_time:.2f}s.")
                # <<< END ADDED >>>

                # Keep overlapped_frames calculation as it likely relates to soft_append_bcthw
                overlapped_frames = latent_window_size * 4 - 3
                # <<< MODIFIED: Use potentially corrected pixels for append >>>
                history_pixels = soft_append_bcthw(pixels_for_append, history_pixels, overlapped_frames)
                # <<< END MODIFIED >>>

            logger.info(f"{section_progress_prefix}After VAE decode/append. History pixels shape: {history_pixels.shape}") # W_LOG 12
            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id_prefix}{worker_job_id[:8]}.mp4')

            logger.info(f"{section_progress_prefix}Saving MP4 to {output_filename}...") # W_LOG 13
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            logger.info(f'{section_progress_prefix}Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            # Update TaskManager state with the saved file path
            update_state({'last_video_path': output_filename})

            if is_last_section:
                break

    except SkipTaskException:
        exception_occurred = True # Set flag
        logger.warning(f"Task skipped.")
        # Update state to reflect skip/cancel
        update_state({'status': 'Skipped/Cancelled', 'is_finished': True, 'is_running': False, 'is_cancelled': True})
        return ('skipped', {})
    except KeyboardInterrupt: # Catch explicit cancel signal from callback
        exception_occurred = True # Set flag
        logger.warning(f"Task cancelled by stop request.")
        update_state({'status': 'Cancelled by user', 'is_finished': True, 'is_running': False, 'is_cancelled': True})
        return ('cancel', {})
    except Exception as e:
        exception_occurred = True # Set flag
        error_message = f"Unhandled exception in worker: {e}"
        logger.exception(error_message) # Log exception with traceback
        # Update state with error
        update_state({'status': f'Error: {error_message}', 'error_message': error_message, 'is_finished': True, 'is_running': False})
        return ('error', {'error': error_message})
    finally:
        # Final cleanup regardless of success or failure
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    # --- Handle Normal Finish --- #
    if not exception_occurred:
        logger.info(f"Worker finished task successfully.")
        # Update final state for success
        final_status = 'Finished Successfully' # Store status text
        final_state = {
            'status': final_status,
            'is_finished': True,
            'is_running': False,
            'progress_percentage': 100
        }
        if output_filename:
            final_state['last_video_path'] = output_filename
        else:
            logger.warning("Worker finished successfully but output_filename was not set.")
            final_status = 'Finished (Warning: Output filename missing)'
            final_state['status'] = final_status

        update_state(final_state)
        # <<< MODIFIED: Return status and data dictionary >>>
        return ('success', {'last_video_path': output_filename if output_filename else None})
        # <<< END MODIFICATION >>>

    # Worker function just returns now, state is handled via update_state
    # <<< REMOVED: Original implicit return None is replaced by explicit returns above and in exception handlers >>>
    # return

def process(input_image, last_frame_image, # Added last_frame_image
            # <<< ADDED: New inputs for prompt source >>>
            prompt_source, api_key, base_instruction,
            # <<< END ADDED >>>
            prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, schedule_type, skip_last_n_sections,
            injection_strength,
            # <<< ADDED: Missing parameter for in-loop correction >>>
            correct_in_loop):
            # <<< END ADDED >>>
            # enable_latent_injection, # REMOVED
            # switch_time_percentage): # REMOVED

    # <<< ADDED: Initialize logger for potential Gemini call >>>
    # Use a simple logger for single process messages
    single_logger = logging.getLogger("SingleProcess")
    if not single_logger.hasHandlers():
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s - %(levelname)s - [SingleProcess] %(message)s')
         handler.setFormatter(formatter)
         single_logger.addHandler(handler)
         single_logger.setLevel(logging.INFO)
    # <<< END ADDED >>>

    if input_image is None:
        gr.Warning("No input image (First Frame)!")
        # Return updates for the UI elements this function now controls directly:
        # job_id_state, start_button, end_button
        # We need to add a job_id_state output to the start_button click later
        return None, gr.update(interactive=True), gr.update(interactive=False)

    final_prompt = prompt # Default to manual prompt

    # --- Gemini Prompt Generation Logic --- START ---
    if prompt_source == "Gemini API Generated":
        single_logger.info("Prompt source set to Gemini API.")
        if not api_key:
            gr.Error("Gemini API Key is missing!")
            return None, gr.update(interactive=True), gr.update(interactive=False)
        if not base_instruction:
            gr.Error("Gemini Base Instruction is missing!")
            return None, gr.update(interactive=True), gr.update(interactive=False)

        try:
            # Convert NumPy input image to PIL Image for API call
            input_image_pil = Image.fromarray(input_image)
            single_logger.info("Calling Gemini API to generate prompt...")

            # <<< MODIFIED: Check for last frame image and pass if available >>>
            last_image_pil_for_api = None
            if last_frame_image is not None:
                try:
                    last_image_pil_for_api = Image.fromarray(last_frame_image)
                    single_logger.info("Last frame image provided, will send both to Gemini.")
                except Exception as pil_conv_err:
                    single_logger.warning(f"Could not convert last_frame_image to PIL: {pil_conv_err}. Sending only first frame.")
                    last_image_pil_for_api = None # Ensure it's None on error
            else:
                 single_logger.info("Last frame image not provided, sending only first frame to Gemini.")

            # Call the Gemini API function (pass logger and potentially the last image)
            generated_prompt = call_gemini_api(input_image_pil, base_instruction, api_key, single_logger,
                                               last_image_pil=last_image_pil_for_api)
            # <<< END MODIFIED >>>

            if generated_prompt.startswith("Error:"):
                single_logger.error(f"Gemini API Error: {generated_prompt}")
                gr.Error(f"Gemini API Error: {generated_prompt}")
                return None, gr.update(interactive=True), gr.update(interactive=False)
            else:
                final_prompt = generated_prompt
                single_logger.info(f"Gemini generated prompt successfully: '{final_prompt[:100]}...'")
                # Update the prompt text box to show the generated prompt
                # This requires prompt textbox to be added to outputs of start_button.click
                # For now, we just use final_prompt internally.

        except Exception as e:
            single_logger.exception(f"Error during Gemini API call: {e}")
            gr.Error(f"Error calling Gemini API: {e}")
            # <<< MODIFIED: Return update for prompt as well (no change on error) >>>
            return None, gr.update(interactive=True), gr.update(interactive=False), gr.update()
    # --- Gemini Prompt Generation Logic --- END ---

    # <<< ADDED: Prepare prompt update value >>>
    prompt_update_value = gr.update()
    if prompt_source == "Gemini API Generated" and 'final_prompt' in locals() and not final_prompt.startswith("Error:"):
        # Update prompt box only if Gemini was used and succeeded
        prompt_update_value = gr.update(value=final_prompt)
    # <<< END ADDED >>>

    # Handle Random Seed
    if seed == -1:
        actual_seed = random.randint(0, 2**32 - 1)
        single_logger.info(f"Input seed is -1, using random seed: {actual_seed}")
    else:
        # <<< MODIFIED: Handle empty string or None seed input >>>
        if seed is None or str(seed).strip() == '':
            single_logger.warning(f"Seed input was empty, using random seed.")
            actual_seed = random.randint(0, 2**32 - 1)
            single_logger.info(f"Using random seed: {actual_seed}")
        else:
            try:
                actual_seed = int(seed)
            except (ValueError, TypeError): # Catch TypeError for None as well
                single_logger.warning(f"Invalid seed value '{seed}' provided (type: {type(seed)}). Using random seed.")
                actual_seed = random.randint(0, 2**32 - 1)
                single_logger.info(f"Using random seed: {actual_seed}")
        # <<< END MODIFICATION >>>

    # --- Save Settings --- # (Keep this)
    settings_to_save = {
        'prompt_source': prompt_source, # Added
        'api_key': api_key,             # Added (Security Warning!)
        'base_instruction': base_instruction, # Added
        'prompt': prompt, # Save last used manual prompt (or Gemini output shown)
        'n_prompt': n_prompt,
        'seed': seed,
        'total_second_length': total_second_length,
        'steps': steps,
        'gs': gs,
        'gpu_memory_preservation': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'mp4_crf': mp4_crf,
        'schedule_type': schedule_type,
        'skip_last_n_sections': skip_last_n_sections,
        'injection_strength': injection_strength # Kept
    }
    save_settings(settings_to_save, 'single')
    # ------------------- #

    # --- Collate parameters for TaskManager --- #
    single_params = {
        'input_image_np': input_image,
        'last_frame_image_np': last_frame_image,
        'prompt': final_prompt, # <<< Use the final determined prompt
        'n_prompt': n_prompt,
        'seed': actual_seed,
        'total_second_length': total_second_length,
        'latent_window_size': latent_window_size,
        'steps': steps,
        'cfg': cfg, 'gs': gs, 'rs': rs,
        'gpu_memory_preservation': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'mp4_crf': mp4_crf,
        'schedule_type': schedule_type,
        'skip_last_n_sections': skip_last_n_sections,
        'injection_strength': injection_strength, # Kept strength slider
        # <<< ADDED: Include in-loop flag >>>
        'correct_in_loop': correct_in_loop
        # <<< END ADDED >>>
    }

    # --- Start job via TaskManager --- #
    try:
        job_id = task_manager.start_single_job(single_params)
        print(f"[Single Process] Started job with ID: {job_id}")
    except Exception as e:
        print(f"Error starting single job: {e}")
        traceback.print_exc()
        gr.Error(f"Error starting job: {e}")
        # Return job_id (None) + button updates + prompt update (no change)
        return None, gr.update(interactive=True), gr.update(interactive=False), gr.update()

    # --- Return job_id and initial button states --- #
    # The actual UI updates (video, progress) will be handled by polling
    # <<< MODIFIED: Add prompt_update_value to return tuple >>>
    return job_id, gr.update(interactive=False), gr.update(interactive=True), prompt_update_value


def end_process(job_id): # Now accepts job_id
    # Removed stream logic
    if job_id:
        print(f"[UI Single] Requesting stop for job {job_id}")
        task_manager.request_stop_job(job_id)
        # Optionally return immediate UI feedback (e.g., disable button)
        return gr.update(interactive=False)
    else:
        print("[UI Single] Stop clicked but no active job ID found.")
        return gr.update() # No change


batch_stream = None

@torch.no_grad()
def batch_worker(task_manager, job_id, input_queue, log_file_path, # Added log_file_path
                 input_dir, order, prompt_source, base_instruction, api_key,
                 batch_n_prompt, batch_seed, batch_total_second_length,
                 latent_window_size, batch_steps, cfg, gs, rs,
                 batch_gpu_memory_preservation, batch_use_teacache,
                 batch_mp4_crf, batch_schedule_type, batch_skip_last_n_sections,
                 apply_reinhard_batch,
                 correct_in_loop
                 # ------------------------------------------------- #
                 ):

    # --- Configure Logger --- #
    # <<< MOVED: Get logger instance FIRST >>>
    logger = logging.getLogger(f"batch_job_{job_id}")
    logger.setLevel(logging.INFO) # Set desired logging level

    # Prevent adding multiple handlers if retried/reloaded
    if not logger.hasHandlers():
        # File handler
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        # Console handler (optional, for simultaneous console output)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    # <<< MOVED: Log received correction flag AFTER logger is configured >>>
    logger.info(f"[Debug BW Start {job_id[:8]}] Batch worker started. Log Path: {log_file_path}. Apply Reinhard Correction Flag: {apply_reinhard_batch}")
    # <<< END MOVED >>>
    # ------------------------ #

    bw_id = job_id[:8] # Short ID for logging

    in_q = input_queue
    def update_state(updates):
        task_manager.update_job_state(job_id, updates)

    def check_cancellation_signal(caller_id):
        """Checks the input queue for 'end' or 'skip' signals."""
        logger.info(f"[Debug BW CheckEnter {caller_id}] Checking queue. Top is: {in_q.top()}") # Log entry and top() result
        signal = in_q.top() # Peek at the queue
        if signal == 'end':
            logger.warning(f"[Debug BW Check {caller_id}] Cancel signal ('end') detected. Popping queue.")
            # Remove the signal if it's 'end'
            in_q.pop()
            logger.info(f"[Debug BW CheckExit {caller_id}] Returning True (Cancel)")
            return True # Indicate cancellation requested
        elif signal == 'skip':
             logger.info(f"[Debug BW CheckExit {caller_id}] Returning False (Skip signal found, handled by worker)")
             return False # Don't trigger cancel, let worker handle skip
        else:
             logger.info(f"[Debug BW CheckExit {caller_id}] Returning False (No cancel/skip signal)")
             return False # No cancellation requested

    logger.info(f"[Debug BW Start {bw_id}] Batch worker started. Log Path: {log_file_path}")
    update_state({'status': 'Scanning directory...', 'is_running': True})

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
    image_files = []
    try:
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        if order == 'Sequential':
            image_files.sort()
        logger.info(f"Found {len(image_files)} image files.")
    except Exception as scan_exc:
        err_msg = f"Error scanning directory {input_dir}: {scan_exc}"
        logger.error(err_msg)
        update_state({'status': err_msg, 'is_running': False, 'is_finished': True, 'error_message': err_msg})
        return

    if not image_files:
        err_msg = f"Error: No image files found in {input_dir}"
        logger.error(err_msg)
        update_state({'status': err_msg, 'is_running': False, 'is_finished': True, 'error_message': err_msg})
        return

    job_queue = []
    logger.info(f"Preparing job queue...")
    update_state({'status': f"Found {len(image_files)} images. Preparing job queue..."})

    valid_jobs_count = 0
    skipped_jobs_count = 0
    gemini_prompts_generated = {} # Store generated prompts to avoid re-querying if needed later

    # --- Queue Preparation Loop ---
    for i, img_path in enumerate(image_files):
        logger.info(f"[Debug BW QueuePrep {bw_id}] Preparing job {i+1}/{len(image_files)} for {os.path.basename(img_path)}")
        if check_cancellation_signal(bw_id): # Check for early cancellation
            logger.warning("Batch processing cancelled during queue preparation.")
            update_state({'status': "Batch processing cancelled during queue preparation.", 'is_running': False, 'is_cancelled': True})
            return

        prompt = None
        skip_reason = None

        if prompt_source == "同名 TXT 文件":
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(input_dir, base_name + '.txt')
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    if not prompt:
                        skip_reason = f"Empty prompt file {os.path.basename(txt_path)}"
                except Exception as e:
                    skip_reason = f"Error reading prompt file {os.path.basename(txt_path)} - {e}"
            else:
                 skip_reason = f"Corresponding prompt file {os.path.basename(txt_path)} not found"

        elif prompt_source == "Gemini API 生成":
            # We will generate the prompt *inside* the main processing loop
            # Add to queue with None prompt for now
             prompt = None # Placeholder
        else:
             err_msg = f"Error: Unknown prompt source '{prompt_source}'"
             logger.error(err_msg)
             update_state({'status': err_msg, 'is_running': False, 'is_finished': True, 'error_message': err_msg})
             return

        if skip_reason:
            logger.warning(f"Skipping {os.path.basename(img_path)}: {skip_reason}")
            skipped_jobs_count += 1
        else:
            job_queue.append((img_path, prompt)) # Append even if prompt is None (for Gemini)
            valid_jobs_count += 1

    status_message = f"Queue prepared: {valid_jobs_count} valid jobs, {skipped_jobs_count} skipped."
    logger.info(status_message)
    if valid_jobs_count == 0:
        status_message += f" Error: No valid jobs to process."
        logger.error(status_message)
        update_state({'status': status_message, 'is_running': False, 'is_finished': True, 'error_message': "No valid jobs to process"})
        return
    update_state({'status': status_message})

    # Apply random order after filtering valid jobs
    if order == 'Random':
        random.shuffle(job_queue)
        logger.info("Job queue shuffled.")
        update_state({'status': status_message + " (Shuffled)"})

    # Store final job queue paths in state for UI preview
    job_queue_paths = [p[0] for p in job_queue]
    update_state({'job_queue_paths': job_queue_paths})


    logger.info(f"Starting queue processing ({valid_jobs_count} jobs)..." )
    update_state({'status': f"Starting queue processing ({valid_jobs_count} jobs)..."})

    success_count = 0
    skip_count = skipped_jobs_count # Start with skips from queue prep
    error_count = 0
    processed_images = []
    skipped_images = [] # Track skipped images specifically
    error_details = [] # Store details about errors

    job_finished_normally = True # Assume normal finish unless exception or cancellation occurs

    logger.info(f"[Debug BW {bw_id}] Entering main processing FOR loop. Queue length: {len(job_queue)}")

    try: # Wrap the main loop
        for i, (image_path, initial_prompt) in enumerate(job_queue):
            current_task_id = f"task_{i+1}_{os.path.basename(image_path)[:10]}" # More descriptive ID
            logger.info(f"--- [Debug BW-Loop Start {bw_id}] Iteration {i+1}/{valid_jobs_count}. Image: {os.path.basename(image_path)} ---")
            update_state({'current_job_index': i, 'current_image_path': image_path}) # Update index and path

            logger.info(f"[Debug BW-Loop PreCheck {bw_id}] About to check cancellation signal at start of loop for i={i}") # Log before calling
            # --- Cancellation Check (Start of Loop) ---
            if check_cancellation_signal(bw_id): # Check for early cancellation
                # This logger.warning might not be reached if check_cancellation_signal logs and returns True first
                logger.warning(f"[Debug BW-Loop Cancel {bw_id}] Cancellation signal detected by check. Breaking loop for i={i}.")
                job_finished_normally = False
                break # Keep the conditional break inside the if block
            # The erroneous 'break' statement that was on the next line has been removed.
            logger.info(f"[Debug BW-Loop PostCheck {bw_id}] Passed cancellation check at start of loop for i={i}") # Log after passing check

            # --- Load Image ---
            logger.info(f"[Debug BW-Load {bw_id}] Loading image for i={i}: {image_path}")
            try:
                input_image_pil = Image.open(image_path).convert('RGB')
                input_image_np = np.array(input_image_pil)
            except Exception as e:
                error_msg = f"Error loading image {os.path.basename(image_path)}: {e}"
                logger.error(f"[Debug BW-Load Error {bw_id}] {error_msg}")
                error_count += 1
                error_details.append({'image': image_path, 'error': error_msg})
                update_state({'status': f"Error loading image {os.path.basename(image_path)}."}) # Update simple status
                continue # Skip to next image

            # --- Determine Prompt ---
            current_prompt = initial_prompt
            update_state({'current_prompt': None}) # Clear previous prompt display
            if prompt_source == "Gemini API 生成":
                logger.info(f"[Debug BW-Gemini {bw_id}] Requesting prompt from Gemini API for i={i}")
                update_state({'status': f'Generating prompt for {os.path.basename(image_path)}...'})
                if not base_instruction:
                     logger.warning(f"[Debug BW-Gemini Skip {bw_id}] Skipping i={i}: Base instruction for Gemini API is empty.")
                     skip_count += 1
                     skipped_images.append(image_path)
                     continue
                if not api_key:
                     logger.warning(f"[Debug BW-Gemini Skip {bw_id}] Skipping i={i}: Gemini API Key is not provided.")
                     skip_count += 1
                     skipped_images.append(image_path)
                     continue

                # Call API (Pass logger)
                api_generated_prompt = call_gemini_api(input_image_pil, base_instruction, api_key, logger)
                gemini_prompts_generated[image_path] = api_generated_prompt # Store result regardless of success

                if not api_generated_prompt or api_generated_prompt.startswith("Error:"):
                    logger.error(f"[Debug BW-Gemini Error {bw_id}] Skipping i={i}: Failed to get valid prompt from Gemini API. Response: {api_generated_prompt}")
                    error_count += 1 # Count as error if API fails
                    error_details.append({'image': image_path, 'error': f"Gemini API Error: {api_generated_prompt}"})
                    update_state({'status': f"Error getting prompt for {os.path.basename(image_path)}."})
                    continue
                else:
                    current_prompt = api_generated_prompt
                    logger.info(f"[Debug BW-Gemini Success {bw_id}] Received prompt for i={i}: '{current_prompt[:100]}...'")
                    update_state({'current_prompt': current_prompt}) # Update UI with generated prompt
            else: # Using TXT file prompt
                 if not current_prompt: # Should not happen if queue prep was correct, but check anyway
                     logger.error(f"[Debug BW-Prompt Error {bw_id}] Skipping i={i}: Prompt from TXT file was unexpectedly empty.")
                     error_count += 1
                     error_details.append({'image': image_path, 'error': 'Empty prompt from TXT file after queue prep'})
                     continue
                 logger.info(f"[Debug BW-Prompt TXT {bw_id}] Using prompt from TXT for i={i}: '{current_prompt[:100]}...'")
                 update_state({'current_prompt': current_prompt}) # Update UI with TXT prompt

            # --- Determine Seed ---
            if batch_seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
                seed_info = f"random seed: {current_seed}"
            else:
                current_seed = int(batch_seed) + i # Optionally increment seed per job
                seed_info = f"batch seed (+{i}): {current_seed}"
            logger.info(f"[Debug BW-Seed {bw_id}] Using seed for i={i}: {seed_info}")

            # --- Worker Call Check ---
            logger.info(f"[Debug BW-Loop PreWorkerCheck {bw_id}] About to check cancellation signal before worker call for i={i}") # Log before calling
            if check_cancellation_signal(bw_id): # <-- Check again
                logger.warning(f"[Debug BW-Loop Cancel PreWorker {bw_id}] Cancellation signal received before worker call for i={i}. Breaking loop.")
                job_finished_normally = False
                break
            logger.info(f"[Debug BW-Loop PostWorkerCheck {bw_id}] Passed cancellation check before worker call for i={i}") # Log after passing check

            # --- Worker Call ---
            worker_result = None # Initialize worker_result
            logger.info(f"[Debug BW-Worker Pre {bw_id}] >>> Calling worker for i={i}, image: {image_path}")
            worker_start_time = time.time()
            try:
                # *** CRITICAL: Create a NEW input queue for the worker ***
                # This prevents the batch worker's 'end' signal from stopping the sub-worker prematurely
                # The TaskManager already does this for start_single_job, mimic it here.
                worker_input_queue = FIFOQueue()

                worker_result = worker(
                    task_manager=task_manager, # Pass task_manager
                    job_id=f"{job_id}-{current_task_id}", # Unique ID for sub-job state updates (if worker uses it)
                    logger=logger, # Pass logger
                    main_job_id=job_id, # <<< ADDED main_job_id parameter
                    input_image_np=input_image_np,
                    last_frame_image_np=input_image_np, # Use first frame as last frame for batch
                    prompt=current_prompt,
                    n_prompt=batch_n_prompt,
                    seed=current_seed,
                    total_second_length=batch_total_second_length,
                    latent_window_size=latent_window_size,
                    steps=batch_steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    gpu_memory_preservation=batch_gpu_memory_preservation,
                    use_teacache=batch_use_teacache,
                    mp4_crf=batch_mp4_crf,
                    schedule_type=batch_schedule_type,
                    skip_last_n_sections=batch_skip_last_n_sections,
                    injection_strength=0.0, # Disable injection for now in batch (use first frame as last)
                    # <<< ADDED: Pass the in-loop correction flag >>>
                    correct_in_loop=correct_in_loop,
                    # <<< END ADDED >>>
                    input_queue=worker_input_queue, # *** Pass the NEW queue ***
                    job_id_prefix=f"batch_{job_id[:6]}_{current_task_id}_", # Prefix for worker output files
                    progress_desc_prefix=f"Batch {job_id[:6]} Task {i+1}/{valid_jobs_count}: " # Prefix for worker status updates
                )
                worker_end_time = time.time()
                logger.info(f"[Debug BW-Worker Post {bw_id}] <<< Worker call finished for i={i}. Duration: {worker_end_time - worker_start_time:.2f}s. Raw Result Type: {type(worker_result)}") # Log type first
                # Be cautious logging the full result if it can be large
                if isinstance(worker_result, tuple) and len(worker_result) == 2:
                    # Safely access elements if it's a tuple of length 2
                    status_val = worker_result[0]
                    data_val = worker_result[1]
                    data_info = list(data_val.keys()) if isinstance(data_val, dict) else type(data_val)
                    logger.info(f"[Debug BW-Worker Post Detail {bw_id}] Worker Result: Status={status_val}, Data Info={data_info}")
                elif worker_result is None:
                     # Handle case where worker might return None on success/completion without explicit status
                     logger.info(f"[Debug BW-Worker Post None {bw_id}] Worker returned None, assuming success for i={i}.")
                     worker_result = ('success', {}) # Treat None as success
                else:
                     logger.warning(f"[Debug BW-Worker Post Invalid {bw_id}] Worker Result has unexpected format: {worker_result}")
                     # Force error state if format is wrong
                     worker_result = ('error', {'error': 'Invalid worker result format received', 'details': str(worker_result)})


            except Exception as worker_exc:
                # This catches exceptions *during* the worker call itself
                worker_end_time = time.time()
                logger.error(f"[Debug BW-Worker Exc {bw_id}] !!! Exception DURING worker call for i={i}, image: {image_path}. Duration before exc: {worker_end_time - worker_start_time:.2f}s", exc_info=True)
                # Ensure worker_result is set to an error state tuple
                worker_status = 'error'
                worker_data = {'error': f'Exception in worker: {str(worker_exc)}', 'traceback': traceback.format_exc()}
                worker_result = (worker_status, worker_data) # Ensure worker_result is defined

            # --- Result Processing ---
            logger.info(f"[Debug BW-Result Pre {bw_id}] >>> About to process worker result for i={i}. Current worker_result type: {type(worker_result)}")
            try:
                # Robust check for valid result format BEFORE unpacking
                if not isinstance(worker_result, tuple) or len(worker_result) != 2:
                     logger.error(f"[Debug BW-Result Error {bw_id}] !!! Invalid worker_result format BEFORE unpacking for i={i}: {worker_result}. Forcing error state.")
                     worker_status = 'error'
                     worker_data = {'error': 'Invalid worker result format received', 'details': str(worker_result)}
                else:
                    # Now safe to unpack
                    worker_status, worker_data = worker_result
                    logger.info(f"[Debug BW-Result Post {bw_id}] <<< Unpacked worker result for i={i}. Status: '{worker_status}', Data Info: {list(worker_data.keys()) if isinstance(worker_data, dict) else type(worker_data)}") # Log keys or type

            except Exception as unpack_exc:
                 # This catches exceptions specifically during unpacking
                 logger.error(f"[Debug BW-Result Exc {bw_id}] !!! Exception DURING result unpacking for i={i}", exc_info=True)
                 worker_status = 'error' # Force error status
                 # Ensure worker_data is a dict even after unpack error
                 worker_data = {'error': 'Exception during result unpacking', 'details': str(unpack_exc)}

            # --- Status Check & Update Counts ---
            logger.info(f"[Debug BW-Status Pre {bw_id}] >>> Checking worker_status ('{worker_status}') for i={i}")
            task_final_status = ""
            if worker_status == 'success':
                logger.info(f"[Debug BW-Status Check {bw_id}] Status is 'success' for i={i}")
                success_count += 1
                processed_images.append(image_path)
                task_final_status = "Success"
                # --- Save metadata on success ---
                if isinstance(worker_data, dict) and 'last_video_path' in worker_data and worker_data['last_video_path']:
                    try:
                        output_video_path = worker_data['last_video_path']
                        metadata_file = output_video_path.replace('.mp4', '.json')
                        metadata = {
                            'source_image': image_path,
                            'prompt': current_prompt,
                            'negative_prompt': batch_n_prompt,
                            'seed': current_seed,
                            'cfg': cfg, 'gs': gs, 'rs': rs,
                            'steps': batch_steps,
                            'latent_window_size': latent_window_size,
                            'total_second_length': batch_total_second_length,
                            'schedule_type': batch_schedule_type,
                            'skip_last_n_sections': batch_skip_last_n_sections,
                            'worker_status': worker_status,
                             # Add Gemini prompt if used
                             'gemini_prompt_used': gemini_prompts_generated.get(image_path) if prompt_source == "Gemini API 生成" else None,
                             # --- ADDED: Record if correction was applied --- #
                             'reinhard_correction_applied': False # Default to False
                             # ----------------------------------------------- #
                        }
                        # --- Optionally Apply Correction Here --- #
                        correction_applied_flag = False
                        # <<< ADDED: Log before checking correction flag and worker status >>>
                        logger.info(f"[Debug BW-Correction Check {bw_id}] Checking apply_reinhard_batch ({apply_reinhard_batch}) and worker_status ('{worker_status}') for i={i}")
                        # <<< END ADDED >>>
                        if apply_reinhard_batch and worker_status == 'success': # Check if batch correction is enabled AND task succeeded
                            logger.info(f"[Debug BW-Correction {bw_id}] Conditions met. Attempting Reinhard correction for i={i}.")
                            # Retrieve the reference image path saved by the worker
                            # current_task_state = task_manager.get_job_state(job_id) # Getting main job state unlikely to have sub-task ref path
                            ref_img_path_from_worker = None

                            # Reconstruct the filename the worker SHOULD have saved based on its job_id_prefix logic
                            worker_job_id_prefix = f"batch_{job_id[:6]}_{current_task_id}_"
                            # The worker uses f'{job_id_prefix}{worker_job_id[:8]}_input.png'
                            # worker_job_id here is f"{job_id}-{current_task_id}"
                            sub_job_id = f"{job_id}-{current_task_id}"
                            potential_ref_img_path = os.path.join(outputs_folder, f'{worker_job_id_prefix}{sub_job_id[:8]}_input.png')
                            # <<< ADDED: Log path reconstruction and existence check >>>
                            logger.info(f"  [Debug BW-Correction Path] Reconstructed ref path: {potential_ref_img_path}")
                            ref_exists = os.path.exists(potential_ref_img_path)
                            logger.info(f"  [Debug BW-Correction Path] Path exists: {ref_exists}")
                            # <<< END ADDED >>>

                            if ref_exists:
                                ref_img_path_from_worker = potential_ref_img_path
                            else:
                                 logger.warning(f"[Debug BW-Correction Ref Fail {bw_id}] Could not find expected reference image path for i={i}: {potential_ref_img_path}. Skipping correction.")

                            if ref_img_path_from_worker:
                                # <<< ADDED: Log before calling correction function >>>
                                logger.info(f"  [Debug BW-Correction Call] Calling apply_reinhard_correction with video '{output_video_path}' and ref '{ref_img_path_from_worker}'")
                                # <<< END ADDED >>>
                                corrected_video_file = apply_reinhard_correction(output_video_path, ref_img_path_from_worker) # No progress bar needed here
                                if corrected_video_file:
                                    logger.info(f"  Reinhard correction successful. Corrected video: {corrected_video_file}")
                                    correction_applied_flag = True
                                else:
                                    logger.error(f"  Reinhard correction failed for {os.path.basename(output_video_path)}.")
                            # else: handled above
                        # Update metadata flag based on actual application
                        metadata['reinhard_correction_applied'] = correction_applied_flag
                        # -------------------------------------- #
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=4, ensure_ascii=False)
                        logger.info(f"Saved metadata for {os.path.basename(image_path)} to {os.path.basename(metadata_file)}")
                    except Exception as meta_exc:
                        logger.warning(f"Could not save metadata for {os.path.basename(image_path)}: {meta_exc}")
                else:
                    logger.warning(f"No 'last_video_path' found in worker_data for successful job {i+1}, cannot save metadata.")


            elif worker_status == 'skipped' or worker_status == 'cancel': # Treat cancel from worker as skipped task
                # This catches skips/cancels originating *within* the worker function
                log_reason = "skipped in worker" if worker_status == 'skipped' else "cancelled in worker"
                logger.info(f"[Debug BW-Status Check {bw_id}] Status is '{log_reason}' for i={i}")
                skip_count += 1
                skipped_images.append(image_path)
                task_final_status = "Skipped/Cancelled"
            elif worker_status == 'error':
                logger.warning(f"[Debug BW-Status Check {bw_id}] Status is 'error' for i={i}")
                error_count += 1
                error_info = worker_data.get('error', 'Unknown error') if isinstance(worker_data, dict) else 'Worker data not a dict'
                traceback_info = worker_data.get('traceback', None) if isinstance(worker_data, dict) else None
                error_details.append({
                    'image': image_path,
                    'error': error_info,
                    'traceback': traceback_info # Include traceback if available
                    })
                task_final_status = f"Error: {error_info[:100]}..." # Truncate long errors
                logger.error(f"Error detail for {os.path.basename(image_path)}: {error_info}")
                if traceback_info:
                     logger.error(f"Traceback:\n{traceback_info}")
            else:
                 logger.error(f"[Debug BW-Status Check {bw_id}] !!! Unknown worker_status '{worker_status}' for i={i}")
                 error_count += 1 # Treat unknown as error
                 error_details.append({'image': image_path, 'error': f'Unknown worker status: {worker_status}'})
                 task_final_status = "Error (Unknown Status)"

            # --- Update Progress ---
            logger.info(f"[Debug BW-Progress Pre {bw_id}] >>> About to update progress after i={i}")
            total_processed_tasks = i + 1 # How many tasks have been attempted
            progress = total_processed_tasks / valid_jobs_count # Progress through valid jobs
            status_summary = f'Processing {total_processed_tasks}/{valid_jobs_count}... (S:{success_count} K:{skip_count} E:{error_count}) - Last: {os.path.basename(image_path)} ({task_final_status})'

            update_state_payload = {
                'status': status_summary,
                'progress': progress,
                'progress_percentage': int(progress * 100)
            }
            # Update last video path ONLY if the worker succeeded and returned one
            if worker_status == 'success' and isinstance(worker_data, dict) and 'last_video_path' in worker_data:
                 update_state_payload['last_video_path'] = worker_data['last_video_path']

            update_state(update_state_payload)
            logger.info(f"[Debug BW-Progress Post {bw_id}] <<< Progress updated after i={i}. Status: {status_summary}")

            # --- Loop End Check ---
            logger.info(f"[Debug BW-Loop End {bw_id}] Reached end of loop for i={i}.")

            # --- Check for cancellation signal AGAIN before next iteration --- #
            logger.info(f"[Debug BW-Loop PreEndCheck {bw_id}] About to check cancellation signal at end of loop for i={i}") # Log before calling
            if check_cancellation_signal(bw_id): # <-- Check again
                logger.warning(f"[Debug BW-Loop End Cancel {bw_id}] Cancellation signal received at end of loop for i={i}. Breaking loop.")
                job_finished_normally = False
                break
            logger.info(f"[Debug BW-Loop PostEndCheck {bw_id}] Passed cancellation check at end of loop for i={i}") # Log after passing check

            logger.info(f"--- [Debug BW-Loop End {bw_id}] Finished iteration {i+1} ---")

    except Exception as main_loop_exc:
        error_message = f"Unhandled exception in batch worker main loop"
        logger.exception(error_message) # Log exception with traceback
        update_state({'status': f"FATAL Error: Batch stopped unexpectedly: {main_loop_exc}", 'is_running': False, 'is_finished': True, 'error_message': str(main_loop_exc)})
        job_finished_normally = False

    finally:
        final_log_prefix = f"[Debug BW Final {bw_id}]"
        logger.info(f"{final_log_prefix} Exited FOR loop / Entered FINALLY block. job_finished_normally={job_finished_normally}")
        final_state_update = {}
        current_state = task_manager.get_job_state(job_id) # Get fresh state

        # Determine final status message
        final_status_msg = ""
        if current_state:
            if current_state.get('is_cancelled'):
                 final_status_msg = f"Batch processing cancelled by user. (S:{success_count} K:{skip_count} E:{error_count})"
                 final_state_update['is_cancelled'] = True
            elif not job_finished_normally and not current_state.get('error_message'):
                 # Generic error if loop broke unexpectedly without specific error set
                 final_status_msg = f"Batch processing stopped unexpectedly. (S:{success_count} K:{skip_count} E:{error_count})"
                 final_state_update['error_message'] = "Unexpected stop or worker exit"
            elif current_state.get('error_message'):
                 # Error message was already set by exception handler
                 final_status_msg = f"Batch finished with error: {current_state.get('error_message')} (S:{success_count} K:{skip_count} E:{error_count})"
            else: # Should be normal completion
                 final_status_msg = f"Batch processing finished. (Success:{success_count} Skipped:{skip_count} Errors:{error_count})"
        else:
             # Fallback if state couldn't be retrieved
             final_status_msg = f"Batch finished (state unavailable). (S:{success_count} K:{skip_count} E:{error_count})"
             logger.error(f"{final_log_prefix} Could not retrieve final job state for {job_id}.")

        logger.info(f"{final_log_prefix} Final Batch Status: {final_status_msg}")
        final_state_update['status'] = final_status_msg
        final_state_update['is_running'] = False
        final_state_update['is_finished'] = True
        # Ensure progress is 100% only if all tasks were attempted without cancellation/fatal error
        if job_finished_normally and not current_state.get('is_cancelled'):
             final_state_update['progress_percentage'] = 100
             final_state_update['progress'] = 1.0
        elif current_state: # Use last known progress if stopped early
             final_state_update['progress_percentage'] = current_state.get('progress_percentage', 0)
             final_state_update['progress'] = current_state.get('progress', 0.0)

        # --- Log final error details ---
        if error_details:
             logger.error(f"{final_log_prefix} --- Error Summary ---")
             for err_detail in error_details:
                 logger.error(f"  Image: {os.path.basename(err_detail['image'])}, Error: {err_detail['error']}")
             logger.error(f"{final_log_prefix} --- End Error Summary ---")


        # --- Apply final state update ---
        if current_state:
            logger.info(f"{final_log_prefix} Applying final state update: {final_state_update}")
            update_state(final_state_update)
        else:
             logger.error(f"{final_log_prefix} Skipping final state update as job state could not be retrieved.")


        logger.info(f"{final_log_prefix} Batch worker function finished execution.")
        # --- Close Logger Handlers --- #
        handlers = logger.handlers[:]
        for handler in handlers:
            try:
                 handler.close()
                 logger.removeHandler(handler)
            except Exception as close_exc:
                 print(f"Error closing logger handler {handler}: {close_exc}") # Use print as logger might be closed
        # --------------------------- #

def process_batch(input_dir, order, prompt_source, base_instruction, api_key,
                  batch_n_prompt, batch_seed, batch_total_second_length,
                  latent_window_size, batch_steps, cfg, gs, rs,
                  batch_gpu_memory_preservation, batch_use_teacache,
                  batch_mp4_crf, batch_schedule_type, batch_skip_last_n_sections,
                  apply_reinhard_batch,
                  correct_in_loop_batch
                 ):

    if not input_dir or not os.path.isdir(input_dir):
        gr.Warning("Please provide a valid input directory.")
        # Outputs: current_job_id_state + 3 buttons
        return None, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)

    if prompt_source == "Gemini API 生成" and not base_instruction:
         gr.Warning("Please provide a base instruction when using Gemini API.")
         return None, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)

    if prompt_source == "Gemini API 生成" and not api_key:
        gr.Warning("Please provide your Gemini API Key.")
        return None, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)

    # --- Collate batch parameters ---
    batch_params = {
        'input_dir': input_dir, 'order': order, 'prompt_source': prompt_source,
        'base_instruction': base_instruction, 'api_key': api_key, 'batch_n_prompt': batch_n_prompt,
        'batch_seed': batch_seed, 'batch_total_second_length': batch_total_second_length,
        'latent_window_size': latent_window_size, 'batch_steps': batch_steps, 'cfg': cfg, 'gs': gs, 'rs': rs,
        'batch_gpu_memory_preservation': batch_gpu_memory_preservation, 'batch_use_teacache': batch_use_teacache,
        'batch_mp4_crf': batch_mp4_crf, 'batch_schedule_type': batch_schedule_type, 'batch_skip_last_n_sections': batch_skip_last_n_sections,
        # --- ADDED: Pass batch correction flag --- #
        'apply_reinhard_batch': apply_reinhard_batch,
        # <<< ADDED: Pass in-loop correction flag --- #
        'correct_in_loop': correct_in_loop_batch
        # <<< END ADDED >>>
        # ----------------------------------------- #
    }
    # --- Save Batch Settings (Keep this as is) --- D
    # Ensure the keys used here match the keys expected by batch_worker and TaskManager
    batch_params_for_save = batch_params.copy() # Make a copy for saving
    # batch_params_for_save['batch_use_teacache'] = batch_params_for_save.pop('use_teacache') # No longer needed to rename here
    save_settings(batch_params_for_save, 'batch')
    # -------------------------

    # --- Start job via TaskManager --- A
    try:
        # Pass the original batch_params which includes 'batch_use_teacache'
        job_id = task_manager.start_new_job(batch_params)
    except Exception as e:
        print(f"Error starting batch job: {e}")
        traceback.print_exc()
        # Return job_id (None) + 3 button updates
        gr.Error(f"Error starting job: {e}") # Show error popup
        # Outputs: job_id_state + 8 batch_ops components (excluding job_id itself)
        return (None,) + (gr.update(interactive=True),) + (gr.update(interactive=False),)*7 # Initial state for 8 ops: start=T, rest=F
    # --------------------------------

    # --- Return initial UI state and the job_id --- #
    # Outputs: job_id_state + 8 batch_ops components
    return (
        job_id,
        gr.update(value=f"Job {job_id[:8]} Queued..."), # batch_status
        gr.update(value=None),                      # current_job_image
        gr.update(value=None),                      # batch_result_video
        gr.update(value=""),                       # current_gemini_prompt_display
        gr.update(value="Queue initializing..."),  # queue_preview_display
        gr.update(value=""),                       # batch_task_progress_bar
        gr.update(interactive=False),             # start_batch_button (Disable)
        gr.update(interactive=True),              # end_batch_button (Enable)
        gr.update(interactive=True)               # skip_button (Enable)
    )

def end_batch_process(job_id): # Accepts job_id
    if job_id:
        print(f"[UI] Requesting stop for job {job_id}")
        task_manager.request_stop_job(job_id)
        # Optionally return immediate UI feedback
        return gr.update(interactive=False) # Disable stop button after click
    else:
        print("[UI] Stop clicked but no active job ID found.")
        return gr.update() # No change

# --- Add Skip Task Function --- #
def skip_current_task_process(job_id):
    if job_id:
        print(f"[UI] Requesting skip for current task in job {job_id}")
        task_manager.request_skip_current_task(job_id)
        # Maybe disable skip button temporarily? Or let polling handle it.
        return gr.update() # Return update for the skip button itself if needed
    else:
        print("[UI] Skip clicked but no active job ID found.")
        return gr.update()
# --------------------------- #

# --- Add Log Tail Reader Function --- #
def read_log_tail(log_file_path, n=50):
    """Reads the last n lines of a log file."""
    if not log_file_path or not os.path.exists(log_file_path):
        return "Log file not found or not yet created."
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            # Simple approach for potentially large files:
            # Read chunks from the end until enough lines are found
            # This avoids loading the entire file into memory.
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            lines = []
            chunk_size = 1024
            num_newlines = 0
            offset = 0

            while num_newlines < n + 1 and offset < file_size:
                offset += chunk_size
                seek_pos = max(0, file_size - offset)
                f.seek(seek_pos)
                chunk = f.read(min(chunk_size, file_size - seek_pos))
                if not chunk:
                    break # Should not happen if offset < file_size
                num_newlines += chunk.count('\n')
                lines.insert(0, chunk)
                if seek_pos == 0:
                    break

            full_text = "".join(lines)
            log_lines = full_text.splitlines()
            return "\n".join(log_lines[-n:])

    except Exception as e:
        return f"Error reading log file: {e}"
# ---------------------------------- #


# --- New Polling Function --- #
def poll_batch_status(job_id):
    """Polls BATCH job status, reads state, and returns updates for BATCH UI components."""

    # <<< ADDED: Get logger instance >>>
    logger = logging.getLogger("poll_batch_status")
    # <<< END ADDED >>>

    # Defaults for BATCH mode
    updates = (
        "Batch Status: Idle", # batch_status
        gr.update(value=None), # current_job_image
        gr.update(value=None), # batch_result_video
        "",                    # current_gemini_prompt_display
        "",                    # queue_preview_display
        "",                    # batch_task_progress_bar
        gr.update(interactive=True), # start_batch_button
        gr.update(interactive=False), # end_batch_button
        gr.update(interactive=False)  # skip_button
    )

    if job_id:
        state = task_manager.get_job_state(job_id)
        if state:
            status_text = state.get('status', 'Unknown status')
            is_running = state.get('is_running', False)
            is_finished = state.get('is_finished', False)
            is_cancelled = state.get('is_cancelled', False)
            error_message = state.get('error_message')
            progress_percentage = state.get('progress_percentage', 0)
            last_video_path = state.get('last_video_path')

            # Determine Button States
            start_btn_interactive = not is_running
            stop_btn_interactive = is_running and not is_finished and not is_cancelled
            skip_btn_interactive = is_running and not is_finished and not is_cancelled # Skip only for batch

            # Generate Progress Bar HTML
            progress_bar_html = ""
            if is_running:
                progress_hint = status_text if status_text not in ['Queued', 'Running...'] else "Processing..."
                progress_bar_html = make_progress_bar_html(progress_percentage, progress_hint)

            # --- Prepare updates for BATCH mode --- #
            batch_status_update = status_text
            current_job_image_update = gr.update(value=None)
            img_path = state.get('current_image_path')
            if is_running and img_path and os.path.exists(img_path):
                try:
                    current_job_image_update = gr.update(value=Image.open(img_path))
                except Exception as e:
                    print(f"[UI Poll Error] Failed to load image {img_path}: {e}")
                    current_job_image_update = gr.update()

            gemini_prompt_update = state.get('current_prompt', '') if state.get('job_params', {}).get('prompt_source') == "Gemini API 生成" else ""

            queue_preview_update = ""
            job_queue_paths = state.get('job_queue_paths')
            current_job_index = state.get('current_job_index', -1)
            preview_limit = 10
            if job_queue_paths and current_job_index >= -1:
                remaining_paths = job_queue_paths[current_job_index + 1:]
                preview_paths = remaining_paths[:preview_limit]
                if preview_paths:
                    queue_preview_update = "\n".join([f"{current_job_index + 2 + j}. {os.path.basename(p)}" for j, p in enumerate(preview_paths)])
                    if len(remaining_paths) > preview_limit:
                        queue_preview_update += f"\n... and {len(remaining_paths) - preview_limit} more"
                elif is_running:
                    queue_preview_update = "Processing last item..."
                else:
                    queue_preview_update = "Queue finished."
            elif is_finished or is_cancelled:
                queue_preview_update = "Queue finished."
            else:
                queue_preview_update = "Queue initializing..."

            updates = (
                batch_status_update,
                current_job_image_update,
                gr.update(value=last_video_path), # batch_result_video
                gemini_prompt_update,
                queue_preview_update,
                progress_bar_html,
                gr.update(interactive=start_btn_interactive),
                gr.update(interactive=stop_btn_interactive),
                gr.update(interactive=skip_btn_interactive)
            )

            # Handle final error display
            # Consider showing gr.Error popup here as well if desired
            # if is_finished and error_message:
            #     gr.Error(f"Job {job_id[:8]} failed: {error_message}")

            # <<< ADDED Debug Logging >>>
            logger.info(f"[Poll Debug {job_id[:8]}] State: run={is_running}, fin={is_finished}, can={is_cancelled}. Buttons: start_int={start_btn_interactive}, stop_int={stop_btn_interactive}, skip_int={skip_btn_interactive}")
            # <<< END Debug Logging >>>

        else:
            # Job ID provided but not found
            updates = list(updates)
            updates[0] = f"Error: Job {job_id[:8]} not found."
            updates = tuple(updates)

    return updates
# -------------------------- #

# --- Add Single Image Polling Function --- #
def poll_single_status(job_id):
    """Polls SINGLE job status, reads state, and returns updates for SINGLE UI components."""

    # Defaults for SINGLE mode (result_video, preview_image, progress_desc, progress_bar, start_button, end_button)
    updates = (
        gr.update(value=None), # result_video
        gr.update(value=None, visible=False), # preview_image (start hidden)
        "Status: Idle",         # progress_desc
        "",                    # progress_bar
        gr.update(interactive=True), # start_button
        gr.update(interactive=False)  # end_button
    )

    if job_id:
        state = task_manager.get_job_state(job_id)
        if state:
            status_text = state.get('status', 'Unknown status')
            is_running = state.get('is_running', False)
            is_finished = state.get('is_finished', False)
            # is_cancelled = state.get('is_cancelled', False) # Cancellation handled by status text
            error_message = state.get('error_message')
            progress_percentage = state.get('progress_percentage', 0)
            last_video_path = state.get('last_video_path')

            # Determine Button States
            start_btn_interactive = not is_running
            stop_btn_interactive = is_running and not is_finished # Consider finished/cancelled state in status_text

            # Generate Progress Bar HTML & Description Text
            progress_bar_html = ""
            progress_desc_text = f"Status: {status_text}"
            if is_running:
                progress_hint = status_text if status_text not in ['Queued', 'Running...'] else "Processing..."
                progress_bar_html = make_progress_bar_html(progress_percentage, progress_hint)
                progress_desc_text = f"Status: {progress_hint} ({progress_percentage}%)"
            elif is_finished and error_message:
                progress_desc_text = f"**Error:** {error_message}\n(Status: {status_text})"
                # Consider adding gr.Error popup maybe?
                # gr.Error(f"Job {job_id[:8]} failed: {error_message}")

            # --- Prepare updates for SINGLE mode --- #
            updates = (
                gr.update(value=last_video_path), # result_video
                gr.update(visible=False),       # preview_image (keep hidden for now)
                progress_desc_text,             # progress_desc
                progress_bar_html,              # progress_bar
                gr.update(interactive=start_btn_interactive),
                gr.update(interactive=stop_btn_interactive)
            )

        else:
            # Job ID provided but not found
            updates = list(updates)
            updates[2] = f"Error: Job {job_id[:8]} not found." # Update progress_desc
            updates = tuple(updates)

    return updates
# -------------------------- #

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()

# Define scheduling strategies globally within the block scope
SCHEDULE_DEFAULT = 'Default Experience Schedule ([3] + [2]*(N-3) + [1, 0])'
SCHEDULE_LINEAR = 'Linear Reversed Schedule (N-1, N-2, ..., 0)'
SCHEDULE_CONSTANT_ZERO = 'Constant Low Padding ([0, 0, ..., 0])' # New
SCHEDULE_START_END_ZERO = 'Start/End Low Padding ([0, 2, ..., 2, 0])' # New
# <<< ADDED: New Anti-Drift Schedule >>>
SCHEDULE_INV_LINEAR_ANTIDRIFT = "Inverted Linear (Anti-Drift Opt) [0, 1, ..., N-1]"
# <<< END ADDED >>>

# --- Remove Hidden State for Current Job Info ---
# current_job_image_state = gr.State(value=None)
# current_job_prompt_state = gr.State(value="")
# -------------------------------------------

# --- Add Function for Single Image Correction --- START
def correct_single_video(job_id, progress=gr.Progress()):
    """Triggers color correction for a completed single image job."""
    if not job_id:
        gr.Warning("没有有效的任务 ID。请先生成一个视频。")
        return None

    state = task_manager.get_job_state(job_id)
    if not state:
        gr.Warning(f"找不到任务 ID {job_id} 的状态信息。")
        return None

    original_video = state.get('last_video_path')
    ref_image = state.get('reference_image_path')

    if not original_video or not os.path.exists(original_video):
        gr.Warning(f"找不到原始视频文件: {original_video}")
        return None
    if not ref_image or not os.path.exists(ref_image):
        gr.Warning(f"找不到用于校正的参考图像文件: {ref_image}")
        return None

    if "_reinhard_corrected" in original_video:
        gr.Info("看起来这个视频已经是校正过的版本了。")
        return original_video # Return existing corrected path

    print(f"[UI Single Correct] Starting Reinhard correction for {job_id}")
    print(f"  Original Video: {original_video}")
    print(f"  Reference Image: {ref_image}")

    # Call the core correction function
    corrected_path = apply_reinhard_correction(original_video, ref_image, progress=progress)

    if corrected_path:
        # Optionally update the main job state? Not strictly necessary.
        # task_manager.update_job_state(job_id, {'corrected_video_path': corrected_path})
        print(f"[UI Single Correct] Correction successful: {corrected_path}")
        # <<< TEMPORARY TEST: Return ORIGINAL video path instead >>>
        print(f"[UI Single Correct TEST] Returning ORIGINAL path: {original_video}")
        return original_video
        # return corrected_path # Original return value commented out
        # <<< END TEMPORARY TEST >>>
    else:
        gr.Error("色彩校正失败，请检查控制台日志。")
        return None
# --- Add Function for Single Image Correction --- END

with block:
    # --- Add State for current Job ID --- A
    current_job_id_state = gr.State(None) # For Batch
    current_single_job_id_state = gr.State(None) # For Single
    # ------------------------------------
    gr.Markdown('# FramePack')

    with gr.Tabs():
        with gr.TabItem("单张生成"): # Single Image Tab
            # Corrected Indentation: The following Row and its content should be INSIDE the TabItem
            with gr.Row(): # <<< INDENTED
                with gr.Column(): # <<< INDENTED
                    input_image = gr.Image(sources='upload', type="numpy", label="Image (First Frame)", height=320)
                    # --- Add Last Frame Input --- #
                    last_frame_image = gr.Image(sources='upload', type="numpy", label="Optional: Image (Last Frame)", height=320)
                    # -------------------------- #

                    # --- Add Prompt Source Selection for Single Image --- START ---
                    prompt_source_single = gr.Radio(
                        label="Prompt Source (Single)",
                        choices=["Manual Input", "Gemini API Generated"],
                        value="Manual Input", # Default to manual
                        elem_id="prompt_source_single_radio" # Add unique ID if needed for JS
                    )
                    with gr.Group(visible=False) as gemini_inputs_group_single: # Start hidden
                        api_key_single = gr.Textbox(
                            label="Gemini API Key (Single)",
                            placeholder="Enter your Gemini API Key",
                            type="password",
                            # Load setting - CAUTION with API keys in settings
                            value=single_settings.get('api_key', "")
                        )
                        base_instruction_single = gr.Textbox(
                            label="Gemini Base Instruction (Single)",
                            placeholder="e.g., Generate a descriptive prompt for video generation based on this image.",
                            lines=2,
                            value=single_settings.get('base_instruction', "")
                        )
                    # --- END Prompt Source Selection ---

                    prompt = gr.Textbox(label="Prompt (Manual Input or Gemini Output)", value='', interactive=True) # Start interactive
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt]) # <<< INDENTED
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False) # <<< INDENTED

                    # --- Add Logic to Show/Hide Gemini Inputs & Toggle Prompt Interactivity --- START ---
                    def toggle_gemini_single(source):
                        is_gemini = source == "Gemini API Generated"
                        return {
                            gemini_inputs_group_single: gr.update(visible=is_gemini),
                            prompt: gr.update(interactive=not is_gemini)
                        }

                    prompt_source_single.change(
                        fn=toggle_gemini_single,
                        inputs=prompt_source_single,
                        outputs=[gemini_inputs_group_single, prompt]
                    )
                    # --- END Logic --- #

                    with gr.Row(): # This row should be inside the Column above # <<< INDENTED
                        start_button = gr.Button(value="Start Generation") # <<< INDENTED
                        end_button = gr.Button(value="End Generation", interactive=False) # <<< INDENTED

                    with gr.Group(): # This group should be inside the Column above # <<< INDENTED
                        # --- Update Schedule Dropdown --- #
                        schedule_type = gr.Dropdown(
                            label="FramePack Scheduling Strategy",
                            choices=[SCHEDULE_DEFAULT, SCHEDULE_LINEAR, SCHEDULE_CONSTANT_ZERO, SCHEDULE_START_END_ZERO,
                                     # <<< ADDED: New schedule to choices >>>
                                     SCHEDULE_INV_LINEAR_ANTIDRIFT
                                     # <<< END ADDED >>>
                                    ],
                            value=single_settings.get('schedule_type', SCHEDULE_DEFAULT),
                            # <<< MODIFIED: Update info text slightly >>>
                            info="Affects video consistency & drift. CONSTANT_ZERO or INV_LINEAR might be better for long videos."
                            # <<< END MODIFIED >>>
                        )
                        # -------------------------------- #
                        # --- Latent Injection Checkbox REMOVED --- #
                        # enable_latent_injection = gr.Checkbox(label="Enable Last Frame Latent Injection", value=single_settings.get('enable_latent_injection', True), info="...")
                        # ----------------------------------------- #
                        # --- Add Injection Strength Slider --- #
                        injection_strength = gr.Slider(label="Last Frame Strength", minimum=0.0, maximum=1.0, value=single_settings.get('injection_strength', 0.7), step=0.05, info="Controls influence of the optional last frame latent (0.0 = None, 1.0 = Full).")
                        # ----------------------------------- #
                        # --- ADDED: In-Loop Correction Checkbox (Single) --- #
                        correct_in_loop_single = gr.Checkbox(label="在生成循环中校正色彩 (实验性)", value=single_settings.get('correct_in_loop', False), info="对解码后的每个片段应用Reinhard校正，可能显著减慢速度并影响效果。")
                        # ------------------------------------------------- #
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=single_settings.get('use_teacache', True), info='Faster speed, but often makes hands and fingers slightly worse.')
                        n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                        seed = gr.Number(label="Seed (-1 for random)", value=single_settings.get('seed', 31337), precision=0)
                        total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=single_settings.get('total_second_length', 5), step=0.1)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=single_settings.get('steps', 25), step=1, info='Changing this value is not recommended.')
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                        gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=single_settings.get('gs', 10.0), step=0.01, info='Changing this value is not recommended.')
                        rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                        gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=single_settings.get('gpu_memory_preservation', 6), step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=single_settings.get('mp4_crf', 16), step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                        skip_last_n_sections = gr.Number(label="Skip Last N Sections", value=single_settings.get('skip_last_n_sections', 0), precision=0, minimum=0, info="Skip generating the final N sections (useful for quick previews).")

                with gr.Column(): # This column should be parallel to the previous Column, inside the outer Row
                    preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    result_video = gr.Video(label="生成结果 (原始)", autoplay=True, show_share_button=False, height=350, loop=True) # Adjusted height
                    # --- Add Correction Button and Output Video --- #
                    with gr.Row():
                        correct_button = gr.Button(value="应用首帧色彩校正 (Reinhard)")
                    corrected_video = gr.Video(label="色彩校正后结果", autoplay=True, show_share_button=False, height=350, loop=True, visible=True) # Start visible, will be updated
                    # --------------------------------------------- #
                    gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
                    progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                    # Define inputs/outputs for single image generation
                    single_ips = [
                        input_image, # 1
                        last_frame_image, # 2
                        # <<< REORDERED: Match process function signature >>>
                        prompt_source_single, # 3
                        api_key_single, # 4
                        base_instruction_single, # 5
                        prompt, # 6
                        n_prompt, # 7
                        seed, # 8
                        total_second_length, # 9
                        latent_window_size, # 10
                        steps, # 11
                        cfg, # 12
                        gs, # 13
                        rs, # 14
                        gpu_memory_preservation, # 15
                        use_teacache, # 16
                        mp4_crf, # 17
                        schedule_type, # 18
                        skip_last_n_sections, # 19
                        injection_strength, # 20
                        correct_in_loop_single # 21 (Moved to end)
                        # <<< END REORDERED >>>
                    ]
                    # Define the list of UI components for single image polling (Original Output + Progress/Buttons)
                    single_ops = [result_video, preview_image, progress_desc, progress_bar, start_button, end_button]

                    # Update start_button click outputs
                    start_button.click(
                        fn=process,
                        inputs=single_ips,
                        # <<< MODIFIED: Add prompt to outputs >>>
                        outputs=[current_single_job_id_state, start_button, end_button, prompt]
                    )
                    # Update end_button click inputs & outputs
                    end_button.click(
                        fn=end_process,
                        inputs=[current_single_job_id_state],
                        outputs=[end_button]
                    )
                    # --- Connect Correction Button --- #
                    correct_button.click(
                        fn=correct_single_video,
                        inputs=[current_single_job_id_state],
                        outputs=[corrected_video]
                    )
                    # ------------------------------- #
            # End of content for TabItem "单张生成"

        with gr.TabItem("批处理"): # Batch Processing Tab # <<< ALIGNED with previous TabItem
            with gr.Row(): # <<< INDENTED under TabItem
                with gr.Column(scale=2): # <<< INDENTED under Row
                    input_dir = gr.Textbox(label="Input Directory", placeholder="Enter path to directory containing images and (.txt files or for API use)", value=batch_settings.get('input_dir', "")) # Load setting
                    order = gr.Radio(label="Processing Order", choices=["Sequential", "Random"], value=batch_settings.get('order', "Sequential")) # Load setting

                    # --- Prompt Source Selection --- B
                    prompt_source = gr.Radio(
                        label="Prompt Source",
                        choices=["同名 TXT 文件", "Gemini API 生成"],
                        value="同名 TXT 文件" # Reset to default, removed loading
                    )
                    # --- Gemini API Specific Inputs --- C
                    # Set visibility based on default value or interaction, not loaded setting
                    with gr.Group(visible=False) as gemini_inputs_group: # Default to hidden
                        api_key_input = gr.Textbox(
                            label="Gemini API Key",
                            placeholder="Enter your Gemini API Key",
                            type="password",
                            value=batch_settings.get('api_key', "") # Load setting (Security Warning!)
                        )
                        base_instruction_input = gr.Textbox(
                            label="Gemini 任务指令 (与图片一同发送)", # Updated Label
                            placeholder="例如：为这张图片生成一段用于视频生成的描述性提示词",
                            lines=2,
                            value=batch_settings.get('base_instruction', "")
                        )
                    # Show/hide Gemini inputs group based on radio selection
                    prompt_source.change(
                        fn=lambda source: gr.update(visible=source == "Gemini API 生成"),
                        inputs=prompt_source,
                        outputs=gemini_inputs_group
                    )
                    # --------------------------------

                    with gr.Row():
                       start_batch_button = gr.Button(value="Start Batch Processing")
                       skip_button = gr.Button(value="Skip Current Task", interactive=False) # Add Skip Button
                       end_batch_button = gr.Button(value="Stop Batch Processing", interactive=False)

                    # --- Batch Generation Parameters (Mirrors Single Image Tab) --- D
                    with gr.Accordion("Batch Generation Parameters", open=False):
                        # Copied from Single Image Tab, ensure unique variable names if needed
                        # but Gradio handles component references okay usually.
                        # We will reuse the variable names and pass them directly.
                        batch_schedule_type = gr.Dropdown(
                            label="FramePack Scheduling Strategy",
                            choices=[SCHEDULE_DEFAULT, SCHEDULE_LINEAR, SCHEDULE_CONSTANT_ZERO, SCHEDULE_START_END_ZERO,
                                     # <<< ADDED: New schedule to choices >>>
                                     SCHEDULE_INV_LINEAR_ANTIDRIFT
                                     # <<< END ADDED >>>
                                     ], # Added new choices
                            value=batch_settings.get('batch_schedule_type', SCHEDULE_DEFAULT),
                            # <<< MODIFIED: Update info text slightly >>>
                            info="Affects video consistency & drift. CONSTANT_ZERO or INV_LINEAR might be better for long videos."
                            # <<< END MODIFIED >>>
                        )
                        batch_use_teacache = gr.Checkbox(label='Use TeaCache', value=batch_settings.get('batch_use_teacache', True), info='Faster speed, slightly worse hands/fingers.')
                        batch_n_prompt = gr.Textbox(label="Negative Prompt (Batch)", value="", visible=False)
                        batch_seed = gr.Number(label="Seed (Batch Start, -1 for random per image)", value=batch_settings.get('batch_seed', 12345), precision=0, info="Set starting seed for the batch, or -1 for a random seed for each image.") # Load setting, Updated label/info
                        batch_total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=batch_settings.get('batch_total_second_length', 5), step=0.1) # Load setting
                        batch_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=batch_settings.get('batch_steps', 25), step=1, info='Changing is not recommended.')
                        batch_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=batch_settings.get('batch_gpu_memory_preservation', 6), step=0.1, info="Larger value means slower speed but less OOM risk.")
                        batch_mp4_crf = gr.Slider(label="MP4 Compression (CRF)", minimum=0, maximum=100, value=batch_settings.get('batch_mp4_crf', 16), step=1, info="Lower means better quality. 0=lossless. Try 16 if outputs are black.")
                        batch_skip_last_n_sections = gr.Number(label="Skip Last N Sections (Batch)", value=batch_settings.get('batch_skip_last_n_sections', 0), precision=0, minimum=0, info="Skip generating the final N sections for each batch item.")
                        # --- Add Batch Correction Checkbox --- #
                        apply_reinhard_batch = gr.Checkbox(label="完成后自动应用首帧色彩校正 (Reinhard)", value=batch_settings.get('apply_reinhard_batch', False), info="对每个成功生成的视频应用Reinhard色彩校正，使用该视频的第一帧作为参考。会额外保存一个校正后的视频文件。")
                        # --- ADDED: In-Loop Correction Checkbox (Batch) --- #
                        correct_in_loop_batch = gr.Checkbox(label="在生成循环中校正色彩 (实验性)", value=batch_settings.get('correct_in_loop_batch', False), info="对解码后的每个片段应用Reinhard校正，可能显著减慢速度并影响效果。")
                        # ------------------------------------------------ #
                        # ----------------------------------- #
                    # --------------------------------------------------------------

                with gr.Column(scale=3):
                    # --- Update UI Components --- #
                    batch_status = gr.Textbox(label="Batch Status", interactive=False, lines=1) # Keep simple status
                    with gr.Row():
                         current_job_image = gr.Image(label="Current Image", height=150, width=150, interactive=False)
                         # Remove current_job_prompt as it's in the log
                         # current_job_prompt = gr.Textbox(label="Current Prompt Used", lines=5, interactive=False)
                         # Add dedicated display for Gemini Prompt
                         current_gemini_prompt_display = gr.Textbox(label="Current Prompt (Gemini Generated)", lines=5, interactive=False)

                    batch_result_video = gr.Video(label="Last Finished Batch Item", height=300)
                    # Remove Log Display, Add Queue Preview Display
                    queue_preview_display = gr.Textbox(label="Queue Preview (Next 10)", lines=5, interactive=False)
                    # Re-add progress bar display
                    batch_task_progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                    # Remove obsolete components
                    # batch_preview_image = gr.Image(label="Current Batch Item Preview", height=200, visible=False)
                    # batch_progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                    # queue_preview = gr.Textbox(label="Queue Preview (Next 10)", lines=5, interactive=False)
                    # -------------------------- #

            # Define inputs/outputs for batch processing
            batch_ips = [
                input_dir, order, prompt_source, base_instruction_input, api_key_input,
                batch_n_prompt, batch_seed, batch_total_second_length,
                latent_window_size, batch_steps, cfg, gs, rs,
                batch_gpu_memory_preservation, batch_use_teacache,
                batch_mp4_crf, batch_schedule_type, batch_skip_last_n_sections,
                # --- ADDED: Pass batch correction flag --- #
                apply_reinhard_batch,
                # <<< ADDED: Pass in-loop correction flag --- #
                correct_in_loop_batch
                # <<< END ADDED >>>
                # ----------------------------------------- #
            ]
            # --- Updated batch_ops for new UI --- #
            # Order must match the return tuple of poll_batch_status when is_batch_mode=True
            batch_ops = [
                batch_status,                   # Index 0
                current_job_image,              # Index 1
                batch_result_video,             # Index 2
                current_gemini_prompt_display,  # Index 3
                queue_preview_display,          # Index 4
                batch_task_progress_bar,        # Index 5 (New)
                start_batch_button,             # Index 6
                end_batch_button,               # Index 7
                skip_button                     # Index 8
            ]
            # ------------------------------------ #

            # --- Adjust start_batch_button click --- #
            # process_batch now returns job_id and initial button states
            # Need to match the number of outputs to the function's return tuple (now 9 values)
            start_batch_button.click(
                fn=process_batch,
                inputs=batch_ips,
                # Outputs: job_id_state + the 8 components in batch_ops
                outputs=[current_job_id_state] + batch_ops
            )

            # Adjust end_batch_button click
            end_batch_button.click(
                fn=end_batch_process,
                inputs=[current_job_id_state],
                outputs=[end_batch_button] # Output to the button itself
            )

            # Add skip_button click event
            skip_button.click(
                fn=skip_current_task_process,
                inputs=[current_job_id_state],
                outputs=[skip_button] # Optionally disable skip button immediately
            )

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    # --- Setup Polling for BOTH Single and Batch --- #
    timer = gr.Timer(3) # Set polling interval (e.g., 3 seconds)

    # Batch Polling (Existing)
    timer.tick(
        fn=poll_batch_status, # Corrected function name
        inputs=[current_job_id_state],
        outputs=batch_ops # Unpack updates to the components
    )
    block.load(
        fn=poll_batch_status, # Corrected function name
        inputs=[current_job_id_state],
        outputs=batch_ops,
    )

    # Single Image Polling (New)
    # Use a slightly different interval to avoid potential simultaneous updates?
    single_timer = gr.Timer(3.1)
    single_timer.tick(
        fn=poll_single_status, # Corrected function name
        inputs=[current_single_job_id_state],
        outputs=single_ops # Unpack updates to single components
    )
    block.load(
        fn=poll_single_status, # Corrected function name
        inputs=[current_single_job_id_state],
        outputs=single_ops
    )
    # ---------------------------------------------- #

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
