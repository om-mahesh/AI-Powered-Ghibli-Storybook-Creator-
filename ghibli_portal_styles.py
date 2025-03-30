import cv2
import numpy as np
from enum import Enum
import random

class ArtisticStyle(Enum):
    GHIBLI = "studio ghibli style, hand-drawn animation, pastel colors, miyazaki"
    CYBERPUNK = "cyberpunk style, neon colors, high tech, dark atmosphere, blade runner"
    WATERCOLOR = "watercolor painting style, soft edges, flowing colors, artistic"
    FANTASY = "fantasy art style, magical, ethereal, detailed, dreamlike"
    STEAMPUNK = "steampunk style, vintage, mechanical, brass, victorian aesthetic"
    UKIYO_E = "ukiyo-e style, japanese woodblock print, traditional"
    IMPRESSIONIST = "impressionist painting style, visible brushstrokes, soft"
    PIXEL_ART = "pixel art style, 8-bit, retro gaming, pixelated"

def get_style_prompt_suffix(style):
    """Get the prompt suffix for a given artistic style"""
    if isinstance(style, ArtisticStyle):
        return style.value
    elif style in [s.name for s in ArtisticStyle]:
        return ArtisticStyle[style].value
    else:
        return ArtisticStyle.GHIBLI.value

class VisualEffect:
    @staticmethod
    def none(frame):
        """No effect, return the original frame"""
        return frame
    
    @staticmethod
    def soft_glow(frame, intensity=0.3):
        """Add a soft glow effect to the frame"""
        # Create a blurred version of the frame
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Blend the original and blurred frame
        result = cv2.addWeighted(frame, 1.0, blur, intensity, 0)
        return result
    
    @staticmethod
    def warm_tone(frame):
        """Apply a warm color tone to the frame"""
        # Increase red and green channels
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shift hue slightly toward orange/yellow
        h = np.mod(h + 10, 180).astype(np.uint8)
        
        # Increase saturation slightly
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        
        # Merge channels and convert back to BGR
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def cool_tone(frame):
        """Apply a cool color tone to the frame"""
        # Increase blue channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shift hue slightly toward blue
        h = np.mod(h - 10, 180).astype(np.uint8)
        
        # Merge channels and convert back to BGR
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def vintage(frame):
        """Apply a vintage film look to the frame"""
        # Convert to float32
        frame_float = frame.astype(np.float32) / 255.0
        
        # Reduce saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        h, s, v = cv2.split(hsv)
        s *= 0.7  # Reduce saturation
        hsv_desaturated = cv2.merge([h, s, v])
        desaturated = cv2.cvtColor(hsv_desaturated, cv2.COLOR_HSV2BGR)
        
        # Add slight sepia tone
        sepia = np.array([0.272, 0.534, 0.131])
        sepia_tone = cv2.transform(desaturated, np.array([[0.393, 0.769, 0.189],
                                                         [0.349, 0.686, 0.168],
                                                         [0.272, 0.534, 0.131]]))
        
        # Add vignette effect
        height, width = frame.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        
        # Create vignette mask
        vignette = np.clip(1 - radius * 0.5, 0, 1)
        vignette = np.dstack([vignette] * 3)
        
        # Apply vignette
        vintage_frame = sepia_tone * vignette
        
        # Add film grain
        grain = np.random.normal(0, 0.03, frame.shape).astype(np.float32)
        vintage_frame = np.clip(vintage_frame + grain, 0, 1)
        
        # Convert back to uint8
        return (vintage_frame * 255).astype(np.uint8)
    
    @staticmethod
    def dream(frame):
        """Apply a dreamy, ethereal effect to the frame"""
        # Apply a strong blur for base
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Blend with original (mainly keep the blur but retain some detail)
        dreamy = cv2.addWeighted(frame, 0.4, blur, 0.6, 0)
        
        # Increase brightness slightly
        brightness = np.ones(dreamy.shape, dtype=np.uint8) * 15
        dreamy = cv2.add(dreamy, brightness)
        
        # Apply a subtle color shift
        hsv = cv2.cvtColor(dreamy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Subtle hue shift
        h = np.mod(h + 5, 180).astype(np.uint8)
        
        # Increase saturation
        s = np.clip(s * 1.3, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        hsv_modified = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def cyberpunk(frame):
        """Apply a cyberpunk-inspired neon effect to the frame"""
        # Increase contrast
        alpha = 1.1  # Contrast control
        frame_float = frame.astype(np.float32) / 255.0
        contrast = np.clip(frame_float * alpha, 0, 1.0)
        
        # Split channels
        b, g, r = cv2.split(contrast)
        
        # Boost blues and pinks/purples
        b = np.clip(b * 1.2, 0, 1.0)
        r = np.clip(r * 1.2, 0, 1.0)
        
        # Merge channels
        enhanced = cv2.merge([b, g, r])
        
        # Convert to HSV for further adjustments
        hsv = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation
        s = np.clip(s.astype(np.int32) + 30, 0, 255).astype(np.uint8)
        
        # Add "bloom" effect to bright areas
        _, highlight_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
        highlight_mask = cv2.dilate(highlight_mask, None, iterations=3)
        
        # Create bloom layer
        bloom = cv2.GaussianBlur(cv2.merge([b*255, g*255, r*255]), (21, 21), 0)
        
        # Merge HSV channels and convert back to BGR
        hsv_enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Apply bloom where highlights were detected
        highlight_mask_3ch = cv2.merge([highlight_mask, highlight_mask, highlight_mask]) / 255.0
        result = cv2.addWeighted(result, 1.0, bloom.astype(np.uint8), 0.3 * (highlight_mask_3ch), 0)
        
        return result
    
    @staticmethod
    def rain_effect(frame):
        """Add a rain effect to the frame"""
        height, width = frame.shape[:2]
        
        # Create a black canvas for rain
        rain_layer = np.zeros_like(frame)
        
        # Generate random rain drops
        for _ in range(300):  # Number of raindrops
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            length = random.randint(5, 15)  # Length of raindrop
            thickness = random.randint(1, 2)  # Thickness of raindrop
            
            # Don't let drops go out of bounds
            if y + length > height:
                length = height - y - 1
                
            # Draw the raindrop as a line
            if length > 0:
                cv2.line(rain_layer, (x, y), (x, y + length), (200, 200, 220), thickness)
        
        # Blur the rain slightly
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        # Darken the original image a bit for rain effect
        darkened = cv2.addWeighted(frame, 0.85, np.zeros_like(frame), 0, 0)
        
        # Add rain to the image
        result = cv2.add(darkened, rain_layer)
        
        return result
    
    @staticmethod
    def get_effect_by_name(effect_name):
        """Get effect function by name"""
        effects = {
            "none": VisualEffect.none,
            "soft_glow": VisualEffect.soft_glow,
            "warm_tone": VisualEffect.warm_tone,
            "cool_tone": VisualEffect.cool_tone,
            "vintage": VisualEffect.vintage,
            "dream": VisualEffect.dream,
            "cyberpunk": VisualEffect.cyberpunk,
            "rain": VisualEffect.rain_effect
        }
        
        return effects.get(effect_name, VisualEffect.none)

# Example of advanced text parsing for user commands
def parse_user_command(command):
    """
    Parse a user command for style and effect changes
    Returns tuple of (base_prompt, style, effect)
    
    Example commands:
    - "forest with mountains, style:cyberpunk, effect:rain"
    - "beach sunset, style:watercolor"
    - "snowy village, effect:vintage"
    """
    # Default values
    style = ArtisticStyle.GHIBLI
    effect = "none"
    
    # Check for style command
    if "style:" in command:
        for s in ArtisticStyle:
            style_tag = f"style:{s.name.lower()}"
            if style_tag in command.lower():
                style = s
                command = command.lower().replace(style_tag, "").strip()
    
    # Check for effect command
    if "effect:" in command:
        for e in ["none", "soft_glow", "warm_tone", "cool_tone", "vintage", "dream", "cyberpunk", "rain"]:
            effect_tag = f"effect:{e}"
            if effect_tag in command.lower():
                effect = e
                command = command.lower().replace(effect_tag, "").strip()
    
    # Clean up the base prompt
    base_prompt = command.strip().rstrip(",").strip()
    
    return (base_prompt, style, effect)

# Function to get available styles and effects for display to user
def get_available_styles_and_effects():
    """Return lists of available styles and effects"""
    styles = [s.name.lower() for s in ArtisticStyle]
    
    effects = ["none", "soft_glow", "warm_tone", "cool_tone", 
               "vintage", "dream", "cyberpunk", "rain"]
    
    return styles, effects 