"""
Linux-style notification popup generator.
Creates a notification-style popup window similar to Linux system notifications/dialogs.
Refined: 850x450 (Larger per user request), Single-word button optimization.
Fixed: Text overlapping/garbled issues using robust text wrapping (handle \\n and \n).
Restored: Icon support (Exclamation Mark).
Updated: Support for dynamic Cancel button label.
"""
from typing import Tuple, Dict, Optional
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import re

def create_linux_notification_popup(
    width: int = 850, # UPDATED: 850
    height: int = 450, # UPDATED: 450
    title: str = "System Update Required",
    message: str = "A critical security patch is pending.\\nPlease authorize to continue.",
    button_label: str = "Authorize",
    cancel_label: str = "Cancel", 
    bg_color: Tuple[int, int, int] = (255, 255, 255),  # White
    text_color: Tuple[int, int, int] = (20, 20, 20),  # Dark Grey
    border_color: Tuple[int, int, int] = (180, 180, 180),  # Light gray
    border_width: int = 1,
    shadow_offset: int = 5,
    shadow_blur: int = 10,
    padding: int = 40,
    title_font_size: int = 28,
    message_font_size: int = 20,
    button_font_size: int = 20,
    icon_size: int = 64,  
    show_icon: bool = True,  
    icon_color: Tuple[int, int, int] = (255, 140, 0),
) -> Tuple[Image.Image, Dict[str, Dict[str, int]]]:
    """
    Create a Linux-style dialog popup image with two buttons.
    """
    # Create image with extra space for shadow
    img_width = width + shadow_offset + shadow_blur
    img_height = height + shadow_offset + shadow_blur
    img = Image.new("RGB", (img_width, img_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw shadow
    for i in range(shadow_blur):
        alpha = int(40 * (1 - i / shadow_blur))
        shadow_rect = [
            shadow_offset + i,
            shadow_offset + i,
            width + shadow_offset - i,
            height + shadow_offset - i
        ]
        shadow_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.rectangle(shadow_rect, fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), shadow_img).convert("RGB")
        draw = ImageDraw.Draw(img)
    
    # Draw main popup rectangle
    popup_x = shadow_offset
    popup_y = shadow_offset
    popup_rect = [popup_x, popup_y, popup_x + width, popup_y + height]
    
    # Draw background
    draw.rectangle(popup_rect, fill=bg_color, outline=border_color, width=border_width)
    
    # Load fonts
    def load_font(name, size):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            try:
                return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
            except (IOError, OSError):
                return ImageFont.load_default()

    title_font = load_font("DejaVuSans-Bold.ttf", title_font_size)
    button_font = load_font("DejaVuSans-Bold.ttf", button_font_size)
    message_font = load_font("DejaVuSans.ttf", message_font_size)
    
    # Draw Icon
    icon_x = popup_x + padding
    icon_y = popup_y + padding
    
    if show_icon:
        # Orange circle
        draw.ellipse([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], fill=icon_color)
        # Exclamation mark (Simple scaling)
        excl_w = icon_size // 6
        excl_h = icon_size // 2
        excl_x = icon_x + (icon_size - excl_w) // 2
        excl_y = icon_y + (icon_size - excl_h) // 2
        draw.rectangle([excl_x, excl_y, excl_x + excl_w, excl_y + excl_h - 10], fill=(255, 255, 255))
        draw.ellipse([excl_x, excl_y + excl_h - 8, excl_x + excl_w, excl_y + excl_h], fill=(255, 255, 255))
            
    # Text Layout
    if show_icon:
        text_start_x = icon_x + icon_size + padding
    else:
        text_start_x = popup_x + padding
        
    text_max_width = width - (text_start_x - popup_x) - padding
    
    # Helper for Text Wrapping
    def get_wrapped_lines(text, font, max_width):
        lines = []
        # Normalize newlines: handle both literal \n (from some LLM outputs) and actual newlines
        path_normalized = text.replace("**", "").replace('\\n', '\n')
        paragraphs = path_normalized.split('\n')
        
        for p in paragraphs:
            # Check if paragraph fits
            if hasattr(draw, "textbbox"):
                w = draw.textbbox((0, 0), p, font=font)[2]
            else:
                w = draw.textsize(p, font=font)[0]
                
            if w <= max_width:
                lines.append(p)
            else:
                # Need to wrap words
                words = p.split(' ')
                current_line = []
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    if hasattr(draw, "textbbox"):
                        tw = draw.textbbox((0, 0), test_line, font=font)[2]
                    else:
                        tw = draw.textsize(test_line, font=font)[0]
                        
                    if tw <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
        return lines

    # Draw Title
    title_y = popup_y + padding
    if show_icon:
        title_y = icon_y + 5
    
    draw.text((text_start_x, title_y), title, fill=text_color, font=title_font)
    
    # Draw Message
    if hasattr(draw, "textbbox"):
        _, _, _, title_bottom = draw.textbbox((text_start_x, title_y), title, font=title_font)
        current_y = title_bottom + 30
    else:
        current_y = title_y + title_font_size + 30
        
    wrapped_msg_lines = get_wrapped_lines(message, message_font, text_max_width)
    line_height = message_font_size + 8 
    
    for line in wrapped_msg_lines:
        draw.text((text_start_x, current_y), line, fill=text_color, font=message_font)
        current_y += line_height

    # Button Area (Bottom)
    button_height = 60
    button_width = 200
    spacing = 40
    
    total_buttons_width = (button_width * 2) + spacing
    buttons_start_x = popup_x + (width - total_buttons_width) // 2
    buttons_y = popup_y + height - button_height - 40 
    
    # 1. Cancel/Negative Button (Left)
    cancel_x = buttons_start_x
    cancel_rect = [cancel_x, buttons_y, cancel_x + button_width, buttons_y + button_height]
    draw.rectangle(cancel_rect, fill=(230, 230, 230), outline=(180, 180, 180))
    
    # Use Dynamic Cancel Label
    cancel_text = cancel_label
    if hasattr(draw, "textbbox"):
        cw = draw.textbbox((0, 0), cancel_text, font=button_font)[2]
        ch = draw.textbbox((0, 0), cancel_text, font=button_font)[3]
    else:
        cw, ch = draw.textsize(cancel_text, font=button_font)
        
    tx = cancel_x + (button_width - cw) // 2
    ty = buttons_y + (button_height - ch) // 2 - 2
    draw.text((tx, ty), cancel_text, fill=(80, 80, 80), font=button_font)
    
    # 2. Action Button (Right)
    action_x = cancel_x + button_width + spacing
    action_rect = [action_x, buttons_y, action_x + button_width, buttons_y + button_height]
    draw.rectangle(action_rect, fill=(0, 100, 220), outline=(0, 80, 200)) # Blue
    
    # Action Label
    display_label = button_label
    
    if hasattr(draw, "textbbox"):
        aw = draw.textbbox((0, 0), display_label, font=button_font)[2]
        ah = draw.textbbox((0, 0), display_label, font=button_font)[3]
    else:
        aw, ah = draw.textsize(display_label, font=button_font)
        
    tx = action_x + (button_width - aw) // 2
    ty = buttons_y + (button_height - ah) // 2 - 2
    draw.text((tx, ty), display_label, fill=(255, 255, 255), font=button_font)
    
    buttons_bbox = {
        "cancel": {
            "xmin": cancel_rect[0], "ymin": cancel_rect[1],
            "xmax": cancel_rect[2], "ymax": cancel_rect[3]
        },
        "action": {
            "xmin": action_rect[0], "ymin": action_rect[1],
            "xmax": action_rect[2], "ymax": action_rect[3]
        }
    }
    
    return img, buttons_bbox


def create_linux_notification_popup_bytes(
    *args,
    **kwargs
) -> Tuple[bytes, Dict[str, Dict[str, int]]]:
    img, buttons_bbox = create_linux_notification_popup(*args, **kwargs)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), buttons_bbox
