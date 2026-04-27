import os
import numpy as np
from PIL import Image

def is_already_transparent(img):
    """Checks if the image already has an alpha channel with transparency."""
    if img.mode == 'RGBA':
        # Check if there are any pixels with alpha < 255
        alpha = np.array(img)[:, :, 3]
        return np.any(alpha < 255)
    return False

def detect_background_color(img_array):
    """Detects the median color from the 1px border of the image."""
    # Grab the top, bottom, left, and right edge pixels
    top_edge = img_array[0, :, :3]
    bottom_edge = img_array[-1, :, :3]
    left_edge = img_array[:, 0, :3]
    right_edge = img_array[:, -1, :3]
    
    # Combine all border pixels into one list
    border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    
    # Find the median color (R, G, B)
    median_color = np.median(border_pixels, axis=0)
    return median_color

def remove_background(img, bg_color, threshold=15):
    """Makes pixels matching the bg_color transparent based on a threshold."""
    img = img.convert("RGBA")
    data = np.array(img)
    
    rgb = data[:, :, :3]
    
    # Calculate Euclidean distance from the background color for each pixel
    dist = np.linalg.norm(rgb - bg_color, axis=2)
    
    # Create a mask where the distance is within the threshold
    mask = dist < threshold
    
    # Set alpha to 0 where mask is True
    data[mask, 3] = 0
    
    return data

def crop_transparent_edges(img_array):
    """Crops the image to the smallest rectangle containing non-transparent pixels."""
    alpha = img_array[:, :, 3]
    
    # Find rows and columns where alpha is not 0
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    # Get the bounding box coordinates
    if not np.any(rows) or not np.any(cols):
        return img_array # Image is completely transparent
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Slice the array (+1 to include the last pixel)
    return img_array[ymin:ymax+1, xmin:xmax+1]

def process_logos(directory="."):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(directory, filename)
            
            with Image.open(filepath) as img:
                print(f"Processing {filename}...")
                
                # 1. Skip if already transparent
                if is_already_transparent(img):
                    print(f"  > Skipping: {filename} already has transparency.")
                    continue
                
                # 2. Convert to numpy and detect background
                img_array = np.array(img.convert("RGB"))
                bg_color = detect_background_color(img_array)
                
                # 3. Make background transparent
                transparent_data = remove_background(img, bg_color)
                
                # 4. Crop edges
                cropped_data = crop_transparent_edges(transparent_data)
                
                # Save the result
                result_img = Image.fromarray(cropped_data, 'RGBA')
                result_img.save(filepath) # Overwrites original
                print(f"  > Success: Removed background and cropped.")

if __name__ == "__main__":
    process_logos()