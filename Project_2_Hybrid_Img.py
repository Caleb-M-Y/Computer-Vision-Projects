import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2
import os

def create_hybrid_image(cat_img, dog_img, sigma_low=5, sigma_high=3, alpha=1.0):
    """
    Create a hybrid image using Gaussian filtering
    
    Parameters:
    - cat_img: Image to apply highpass filter (will show up close)
    - dog_img: Image to apply lowpass filter (will show up far away)
    - sigma_low: Standard deviation for lowpass Gaussian filter
    - sigma_high: Standard deviation for highpass Gaussian filter
    - alpha: Scaling factor for highpass component
    """
    
    # Keep images in color - work with all channels
    cat_color = cat_img.copy()
    dog_color = dog_img.copy()
    
    # Ensure images are the same size
    if cat_color.shape[:2] != dog_color.shape[:2]:
        # Resize cat to match dog's dimensions
        cat_color = cv2.resize(cat_color, (dog_color.shape[1], dog_color.shape[0]))
    
    # Convert to float for processing
    cat_float = cat_color.astype(np.float32) / 255.0
    dog_float = dog_color.astype(np.float32) / 255.0
    
    # Process each color channel separately
    if len(cat_float.shape) == 3:  # Color image
        dog_lowpass = np.zeros_like(dog_float)
        cat_highpass = np.zeros_like(cat_float)
        
        for channel in range(3):  # RGB channels
            # LOWPASS FILTER (Dog image - appears at far viewing distance)
            dog_lowpass[:,:,channel] = gaussian_filter(dog_float[:,:,channel], sigma=sigma_low)
            
            # HIGHPASS FILTER (Cat image - appears at close viewing distance)
            cat_lowpass_channel = gaussian_filter(cat_float[:,:,channel], sigma=sigma_high)
            cat_highpass[:,:,channel] = cat_float[:,:,channel] - cat_lowpass_channel
    else:  # Grayscale fallback
        dog_lowpass = gaussian_filter(dog_float, sigma=sigma_low)
        cat_lowpass = gaussian_filter(cat_float, sigma=sigma_high)
        cat_highpass = cat_float - cat_lowpass
    
    # Create hybrid image
    hybrid = dog_lowpass + alpha * cat_highpass
    
    # Clip values to valid range [0, 1]
    hybrid = np.clip(hybrid, 0, 1)
    
    return dog_lowpass, cat_highpass, hybrid

# Combined function to load images and create hybrid
def hybrid_image():
    try:
        # Import images from folder
        cat_img = mpimg.imread('cat.bmp')
        dog_img = mpimg.imread('dog.bmp')
        
        # Check if they loaded properly
        if cat_img is None or dog_img is None:
            raise ValueError("Images not loaded properly")
        
        # Display the original images to verify they loaded correctly
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display cat image in color
        axes[0].imshow(cat_img)
        axes[0].set_title('Original Cat Image')
        axes[0].axis('off')
        
        # Display dog image in color
        axes[1].imshow(dog_img)
        axes[1].set_title('Original Dog Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print image information
        print(f"Cat image shape: {cat_img.shape}")
        print(f"Dog image shape: {dog_img.shape}")
        print(f"Cat image dtype: {cat_img.dtype}")
        print(f"Dog image dtype: {dog_img.dtype}")
        
        # Create hybrid image with different sigma values
        print("\nCreating hybrid image...")
        dog_low, cat_high, hybrid_result = create_hybrid_image(cat_img, dog_img, 
                                                              sigma_low=8, sigma_high=4, alpha=1.5)
        
        # Display filtering results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filtered components in color
        axes[0,0].imshow(dog_low)
        axes[0,0].set_title('Dog Lowpass Filter (σ=8)')
        axes[0,0].axis('off')
        
        # For highpass, normalize for better visibility
        cat_high_display = (cat_high + 0.5)  # Shift range for better display
        cat_high_display = np.clip(cat_high_display, 0, 1)
        axes[0,1].imshow(cat_high_display)
        axes[0,1].set_title('Cat Highpass Filter (σ=4)')
        axes[0,1].axis('off')
        
        # Final hybrid image in color
        axes[1,0].imshow(hybrid_result)
        axes[1,0].set_title('Hybrid Image (View from far: Dog, View up close: Cat)')
        axes[1,0].axis('off')
        
        # Empty space for layout
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Hybrid image created successfully!")
        print(f"Dog lowpass shape: {dog_low.shape}")
        print(f"Cat highpass shape: {cat_high.shape}")
        print(f"Hybrid image shape: {hybrid_result.shape}")
        
        return cat_img, dog_img, hybrid_result

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None
# Call the combined function
cat_image, dog_image, hybrid_image_result = hybrid_image()

# Show the hybrid image
if hybrid_image_result is not None:
    plt.figure(figsize=(8, 8))
    plt.imshow(hybrid_image_result)
    plt.title('Final Hybrid Image')
    plt.axis('off')
    plt.show()