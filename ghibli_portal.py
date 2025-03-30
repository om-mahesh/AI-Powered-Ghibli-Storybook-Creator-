import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import mediapipe as mp
import time
import os
import argparse
from PIL import Image
import threading
import queue
from ghibli_portal_styles import ArtisticStyle, VisualEffect, get_style_prompt_suffix, parse_user_command, get_available_styles_and_effects
import signal
import platform
import gc

# ----------------------------------------------------
# CONFIGURATIONS
# ----------------------------------------------------
# Model & pipeline settings
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or any SD model on Hugging Face
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image resolution for background generation
BG_WIDTH = 512
BG_HEIGHT = 512

# Segmentation threshold
SEG_THRESHOLD = 0.5

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Folder to save screenshots
SCREENSHOT_DIR = "screenshots"

# Create screenshot directory if it doesn't exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Create folders if don't exist
os.makedirs("fallback", exist_ok=True)

# Generate a basic fallback image and save it
def create_fallback_image():
    """Create a simple gradient image as fallback"""
    fallback_path = os.path.join("fallback", "default_bg.jpg")
    if not os.path.exists(fallback_path):
        # Create a simple gradient background as fallback
        gradient = np.zeros((BG_HEIGHT, BG_WIDTH, 3), dtype=np.uint8)
        for i in range(BG_HEIGHT):
            gradient[i, :, 0] = int(255 * (i / BG_HEIGHT))  # Blue gradient
            gradient[i, :, 1] = int(170 * (1 - i / BG_HEIGHT))  # Green gradient
            gradient[i, :, 2] = 100  # Constant red
        
        # Save the fallback image
        cv2.imwrite(fallback_path, gradient)
        print(f"Created fallback background: {fallback_path}")
    return fallback_path

# Create fallback image at startup
FALLBACK_IMAGE_PATH = create_fallback_image()

# Background generation queue
bg_queue = queue.Queue()
current_bg = None
last_prompt = ""
generating = False
current_style = ArtisticStyle.GHIBLI
current_effect = "none"

# Check if we're on Windows (no SIGALRM)
IS_WINDOWS = platform.system() == "Windows"

class TimeoutError(Exception):
    """Custom exception for timeouts"""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutError("Generation timed out")

def parse_args():
    parser = argparse.ArgumentParser(description="Ghibli-style virtual background portal")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help="Model ID for Stable Diffusion")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device to run the model on (cuda/cpu)")
    parser.add_argument("--width", type=int, default=CAM_WIDTH,
                        help="Camera width")
    parser.add_argument("--height", type=int, default=CAM_HEIGHT,
                        help="Camera height")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible image generation")
    parser.add_argument("--style", type=str, default="ghibli",
                        help="Initial artistic style to use")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps for generation (lower = faster)")
    parser.add_argument("--small", action="store_true",
                        help="Use smaller background size for faster generation")
    parser.add_argument("--ultralight", action="store_true",
                        help="Use extremely light settings for old computers")
    return parser.parse_args()

def generate_background(pipe, prompt, style=ArtisticStyle.GHIBLI, seed=None, steps=30, ultralight=False):
    """Generate a background image using the given prompt and style"""
    global current_bg, generating, current_style
    
    generating = True
    style_suffix = get_style_prompt_suffix(style)
    print(f"Generating background for prompt: '{prompt}' with style: {style.name}")
    
    # If in ultralight mode, don't use SD generation, just create a simple colored background
    if ultralight:
        try:
            # Create a simple colored background based on the prompt
            import hashlib
            # Create a hash from the prompt to get consistent colors for the same prompt
            hash_obj = hashlib.md5(prompt.encode())
            hash_hex = hash_obj.hexdigest()
            
            # Extract color values from the hash
            r = int(hash_hex[0:2], 16)
            g = int(hash_hex[2:4], 16)
            b = int(hash_hex[4:6], 16)
            
            # Create a gradient background
            bg_cv2 = np.zeros((BG_HEIGHT, BG_WIDTH, 3), dtype=np.uint8)
            for i in range(BG_HEIGHT):
                for j in range(BG_WIDTH):
                    # Create a gradient effect
                    factor_i = i / BG_HEIGHT
                    factor_j = j / BG_WIDTH
                    bg_cv2[i, j, 0] = int(b * factor_i)  # Blue channel
                    bg_cv2[i, j, 1] = int(g * (1 - factor_j))  # Green channel
                    bg_cv2[i, j, 2] = int(r * factor_j)  # Red channel
            
            # Resize to match camera feed size
            bg_cv2 = cv2.resize(bg_cv2, (CAM_WIDTH, CAM_HEIGHT))
            
            # Put the background in the queue
            bg_queue.put(bg_cv2)
            print("Created ultralight background (Stable Diffusion disabled)")
            generating = False
            return
        except Exception as e:
            print(f"Error creating ultralight background: {e}")
            # Continue with normal generation if this fails
    
    # Apply seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Use a shorter step count for CPU
    if DEVICE == "cpu" and steps > 15:
        actual_steps = 15
        print(f"Reducing steps to {actual_steps} for better CPU performance")
    else:
        actual_steps = steps
    
    try:
        # Set timeout for generation (60 seconds) - Only for non-Windows systems
        if DEVICE == "cpu" and not IS_WINDOWS:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
        # Silent generation without progress bar
        image = None
        with torch.no_grad():
            with torch.autocast(DEVICE):
                # Add timeout mechanism
                image = pipe(
                    prompt + ", " + style_suffix,
                    height=BG_HEIGHT,
                    width=BG_WIDTH,
                    generator=generator,
                    num_inference_steps=actual_steps
                ).images[0]
                
        # Reset alarm if on non-Windows
        if DEVICE == "cpu" and not IS_WINDOWS:
            signal.alarm(0)
        
        if image:
            # Convert PIL to OpenCV format (numpy array, BGR)
            bg_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Resize to match camera feed size
            bg_cv2 = cv2.resize(bg_cv2, (CAM_WIDTH, CAM_HEIGHT))
            
            # Put the new background in the queue
            bg_queue.put(bg_cv2)
            print("Background generation complete!")
        else:
            raise Exception("No image was generated")
            
    except Exception as e:
        print(f"Error generating background: {e}")
        # Try to load the fallback image first
        try:
            fallback_img = cv2.imread(FALLBACK_IMAGE_PATH)
            if fallback_img is not None:
                # Resize to match camera feed size
                fallback_img = cv2.resize(fallback_img, (CAM_WIDTH, CAM_HEIGHT))
                bg_queue.put(fallback_img)
                print("Using pre-generated fallback background.")
            else:
                raise Exception("Could not load fallback image")
        except:
            # If fallback image loading fails, create a colored background
            bg_cv2 = np.ones((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) * np.array([30, 100, 100], dtype=np.uint8)
            bg_queue.put(bg_cv2)
            print("Using color fallback background due to generation error.")
    
    generating = False

def background_generation_worker(pipe, initial_prompt, initial_style=ArtisticStyle.GHIBLI, seed=None, steps=30, ultralight=False):
    """Worker thread for background generation"""
    global last_prompt, current_style, current_effect
    
    # Generate the initial background
    generate_background(pipe, initial_prompt, initial_style, seed, steps, ultralight)
    
    # Get available styles and effects
    styles, effects = get_available_styles_and_effects()
    
    # Print help message
    print("\nAvailable commands:")
    print("- Enter a new prompt to generate a new background")
    print("- Add 'style:<style_name>' to change artistic style")
    print("- Add 'effect:<effect_name>' to change visual effect")
    print(f"- Available styles: {', '.join(styles)}")
    print(f"- Available effects: {', '.join(effects)}")
    print("- Type 'help' to see this message again")
    print("- Type 'q' to quit")
    
    # Listen for new prompts in a loop
    while True:
        if not bg_queue.empty():
            time.sleep(0.1)  # Prevent busy waiting
            continue
            
        new_command = input("\nEnter a new prompt (or 'help'/'q'): ")
        
        if new_command.lower() == 'q':
            break
        elif new_command.lower() == 'help':
            print("\nAvailable commands:")
            print("- Enter a new prompt to generate a new background")
            print("- Add 'style:<style_name>' to change artistic style")
            print("- Add 'effect:<effect_name>' to change visual effect")
            print(f"- Available styles: {', '.join(styles)}")
            print(f"- Available effects: {', '.join(effects)}")
            print("- Type 'help' to see this message again")
            print("- Type 'q' to quit")
            continue
            
        # Parse the command for prompt, style, and effect
        base_prompt, style, effect = parse_user_command(new_command)
        
        # Update globals
        last_prompt = base_prompt
        current_style = style
        current_effect = effect
        
        # Generate new background if there's a prompt
        if base_prompt:
            generate_background(pipe, base_prompt, style, seed, steps, ultralight)

def apply_beautification_filter(frame):
    """Apply a soft beautification filter to the person in the frame"""
    # Convert to float32
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply slight Gaussian blur for skin smoothing (subtle effect)
    blurred = cv2.GaussianBlur(frame_float, (5, 5), 0)
    
    # Mix original with blurred (maintain some detail)
    result = cv2.addWeighted(frame_float, 0.7, blurred, 0.3, 0)
    
    # Enhance slightly
    result = cv2.multiply(result, 1.1)
    
    # Convert back to uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def add_info_overlay(frame, prompt, style, effect):
    """Add an informative text overlay to the frame"""
    # Create a semi-transparent overlay at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    
    # Add text
    truncated_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
    cv2.putText(overlay, f"Scene: {truncated_prompt}", (20, frame.shape[0] - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(overlay, f"Style: {style.name}", (20, frame.shape[0] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    cv2.putText(overlay, f"Effect: {effect}", (frame.shape[1] - 150, frame.shape[0] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    # Add a status indicator for generation
    status = "Generating..." if generating else "Ready"
    cv2.putText(overlay, status, (frame.shape[1] - 150, frame.shape[0] - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1)
    
    # Blend the overlay with the original frame
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def display_help_overlay(frame):
    """Display keyboard controls help overlay on the frame"""
    # Create a semi-transparent overlay at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    
    # Add help text
    cv2.putText(overlay, "Keyboard Controls:", (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "ESC: Quit | S: Screenshot | B: Beauty filter | H: Hide/Show Help", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(overlay, "Enter new prompts in the console with style:<name> and effect:<name>", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    # Blend the overlay with the original frame
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def main():
    global current_bg, last_prompt, current_style, current_effect
    global BG_WIDTH, BG_HEIGHT  # Make these global for modification
    
    args = parse_args()
    
    # Update global variables based on arguments
    CAM_WIDTH = args.width
    CAM_HEIGHT = args.height
    
    # Reduce background size if small option is selected
    if args.small:
        BG_WIDTH = 320
        BG_HEIGHT = 320
        print("Using smaller background size for faster generation")
    
    # Ultra light mode for very slow computers
    if args.ultralight:
        BG_WIDTH = 256
        BG_HEIGHT = 256
        print("Using ultra-light mode (no AI generation)")
    
    # Set initial style based on argument
    try:
        current_style = ArtisticStyle[args.style.upper()]
    except KeyError:
        print(f"Style '{args.style}' not found, using default Ghibli style")
        current_style = ArtisticStyle.GHIBLI
    
    # Skip loading Stable Diffusion in ultralight mode
    pipe = None
    if not args.ultralight:
        print(f"Initializing on {args.device}...")
        print("Loading Stable Diffusion pipeline. This may take a moment...")
        
        # Initialize the Stable Diffusion pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            revision="fp16" if args.device == "cuda" else None,  # Use fp32 for CPU
            safety_checker=None  # Disable safety checker directly
        )
        pipe.to(args.device)
        
        # Additional optimizations for CPU
        if args.device == "cpu":
            pipe.enable_attention_slicing(slice_size="max")  # More aggressive slicing
            # Disable progress bar which can cause issues
            from diffusers.utils import logging
            logging.set_verbosity_error()
            # Memory management for CPU
            print("Optimizing memory usage for CPU...")
            
            try:
                # Try to enable VAE tiling which helps with memory usage
                pipe.enable_vae_tiling()
                print("VAE tiling enabled for better memory usage")
            except:
                print("Unable to enable VAE tiling - continuing with standard settings")
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Get initial prompt from user
    styles, effects = get_available_styles_and_effects()
    print(f"\nAvailable styles: {', '.join(styles)}")
    print(f"Available effects: {', '.join(effects)}")
    print("You can use these in commands like: 'forest with mountains, style:cyberpunk, effect:rain'")
    
    initial_prompt = input("\nDescribe the environment: ")
    last_prompt = initial_prompt
    
    # Create a function for ultralight background generation
    def create_ultralight_background(prompt, style):
        global current_bg, last_prompt, current_style
        try:
            # Create a simple colored background based on the prompt and style
            import hashlib
            # Create a hash from the prompt and style to get consistent colors
            hash_input = prompt + style.name
            hash_obj = hashlib.md5(hash_input.encode())
            hash_hex = hash_obj.hexdigest()
            
            # Extract color values from the hash
            r = int(hash_hex[0:2], 16)
            g = int(hash_hex[2:4], 16)
            b = int(hash_hex[4:6], 16)
            
            # Create a gradient background
            bg_cv2 = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            for i in range(CAM_HEIGHT):
                for j in range(CAM_WIDTH):
                    # Create a gradient effect
                    factor_i = i / CAM_HEIGHT
                    factor_j = j / CAM_WIDTH
                    bg_cv2[i, j, 0] = int(b * factor_i)  # Blue channel
                    bg_cv2[i, j, 1] = int(g * (1 - factor_j))  # Green channel
                    bg_cv2[i, j, 2] = int(r * factor_j)  # Red channel
            
            return bg_cv2
        except Exception as e:
            print(f"Error creating ultralight background: {e}")
            return np.ones((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) * 100
    
    # Start background generation in a separate thread or directly create ultralight background
    if args.ultralight:
        # Directly create and set the background for ultralight mode
        bg = create_ultralight_background(initial_prompt, current_style)
        bg_queue.put(bg)
        print("Created gradient background in ultralight mode")
        
        # Start a simple thread to handle background changes
        def ultralight_background_worker():
            """Simple worker thread for ultralight mode background generation"""
            global last_prompt, current_style, current_effect
            
            # Get available styles and effects
            styles, effects = get_available_styles_and_effects()
            
            # Print help message
            print("\nAvailable commands:")
            print("- Enter a new prompt to generate a new background")
            print("- Add 'style:<style_name>' to change artistic style")
            print("- Add 'effect:<effect_name>' to change visual effect")
            print(f"- Available styles: {', '.join(styles)}")
            print(f"- Available effects: {', '.join(effects)}")
            print("- Type 'help' to see this message again")
            print("- Type 'q' to quit")
            
            # Listen for new prompts in a loop
            while True:
                if not bg_queue.empty():
                    time.sleep(0.1)  # Prevent busy waiting
                    continue
                    
                new_command = input("\nEnter a new prompt (or 'help'/'q'): ")
                
                if new_command.lower() == 'q':
                    break
                elif new_command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- Enter a new prompt to generate a new background")
                    print("- Add 'style:<style_name>' to change artistic style")
                    print("- Add 'effect:<effect_name>' to change visual effect")
                    print(f"- Available styles: {', '.join(styles)}")
                    print(f"- Available effects: {', '.join(effects)}")
                    print("- Type 'help' to see this message again")
                    print("- Type 'q' to quit")
                    continue
                    
                # Parse the command for prompt, style, and effect
                base_prompt, style, effect = parse_user_command(new_command)
                
                # Update globals
                last_prompt = base_prompt
                current_style = style
                current_effect = effect
                
                # Generate new background if there's a prompt
                if base_prompt:
                    new_bg = create_ultralight_background(base_prompt, style)
                    bg_queue.put(new_bg)
                    print(f"Created new ultralight background for '{base_prompt}'")
        
        # Start the ultralight worker thread
        bg_thread = threading.Thread(
            target=ultralight_background_worker,
            daemon=True
        )
        bg_thread.start()
    else:
        # Start regular background generation in a separate thread
        bg_thread = threading.Thread(
            target=background_generation_worker, 
            args=(pipe, initial_prompt, current_style, args.seed, args.steps, args.ultralight),
            daemon=True
        )
        bg_thread.start()
    
    # Initialize MediaPipe for segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    # Open the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    
    print("\nPortal is running!")
    print("Press 'ESC' to quit")
    print("Press 'S' to save a screenshot")
    print("Press 'B' to toggle beautification filter")
    print("Press 'H' to toggle help overlay")
    print("Enter new prompts in the console to change the background")
    
    # Initialize a placeholder background (gray) until generation completes
    placeholder_bg = np.ones((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) * 80
    current_bg = placeholder_bg
    
    # Flags for various modes
    beautify_enabled = False
    show_help = True
    
    # Main loop
    try:
        while True:
            # Check if there's a new background available
            if not bg_queue.empty():
                current_bg = bg_queue.get()
            
            # Read frame from camera
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera.")
                break
            
            # Mirror the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Apply beautification if enabled
            if beautify_enabled:
                frame = apply_beautification_filter(frame)
            
            # Convert BGR to RGB (for MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform segmentation
            results = segmentation_model.process(frame_rgb)
            mask = results.segmentation_mask
            
            # Create a binary mask and add a little blur for smoother edges
            condition = mask > SEG_THRESHOLD
            mask_blurred = cv2.GaussianBlur((mask * 255).astype(np.uint8), (5, 5), 0) / 255.0
            
            # Ensure background image matches frame size
            bg_resized = cv2.resize(current_bg, (frame.shape[1], frame.shape[0]))
            
            # Use alpha blending for smoother edges
            foreground = frame.astype(float)
            background = bg_resized.astype(float)
            
            # Create a 3-channel mask
            alpha = np.stack((mask_blurred,) * 3, axis=-1)
            
            # Blend foreground and background
            output_frame = foreground * alpha + background * (1 - alpha)
            output_frame = output_frame.astype(np.uint8)
            
            # Apply selected visual effect
            effect_func = VisualEffect.get_effect_by_name(current_effect)
            output_frame = effect_func(output_frame)
            
            # Add text overlay
            output_frame = add_info_overlay(output_frame, last_prompt, current_style, current_effect)
            
            # Add help overlay if enabled
            if show_help:
                output_frame = display_help_overlay(output_frame)
            
            # Display the result
            cv2.imshow("Virtual Portal", output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s') or key == ord('S'):  # Save screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(SCREENSHOT_DIR, f"portal_{timestamp}.jpg")
                cv2.imwrite(filename, output_frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('b') or key == ord('B'):  # Toggle beautification
                beautify_enabled = not beautify_enabled
                print(f"Beautification filter: {'Enabled' if beautify_enabled else 'Disabled'}")
            elif key == ord('h') or key == ord('H'):  # Toggle help overlay
                show_help = not show_help
                print(f"Help overlay: {'Enabled' if show_help else 'Disabled'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Portal closed.")

if __name__ == "__main__":
    main() 