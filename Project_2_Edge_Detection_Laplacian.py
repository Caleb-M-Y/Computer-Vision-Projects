# Laplacian Edge Detection Method

### IMPORTANT!! ###
# this code takes a long time to run because of the nested loops
# The speed of this operation can vary, but please allow it time to run I promise it works

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Project_2_Hybrid_Img import cat_image, dog_image, hybrid_image_result

def laplacian_edge_detection(img, image_name, sigma=1.0):
    """
    Apply Laplacian of Gaussian (LoG) edge detection to an image
    
    Parameters:
    - img: Input image (color or grayscale)
    - image_name: Name for display purposes
    - sigma: Standard deviation for Gaussian blur
    
    Returns:
    - laplacian_result: Laplacian edge detection result
    - zero_crossings: Zero crossings of the Laplacian
    """
    
    # Convert to grayscale for edge detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Convert to uint8 if it's not already (this fixes the hybrid image issue)
    if img.dtype != np.uint8:
        # If it's float in range [0,1], scale to [0,255]
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            # If it's float in range [0,255], just convert
            gray = gray.astype(np.uint8)
    
    # Apply Gaussian blur first
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Take absolute value and normalize
    laplacian_abs = np.absolute(laplacian)
    laplacian_result = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))
    
    # Find zero crossings (sign changes indicate edges)
    # This is a simplified zero crossing detection
    zero_crossings = np.zeros_like(laplacian)
    
    # Check for sign changes in horizontal and vertical directions
    for i in range(1, laplacian.shape[0]-1):
        for j in range(1, laplacian.shape[1]-1):
            # Check if there's a sign change in any direction
            neighbors = [
                laplacian[i-1, j], laplacian[i+1, j],  # vertical neighbors
                laplacian[i, j-1], laplacian[i, j+1],  # horizontal neighbors
                laplacian[i-1, j-1], laplacian[i+1, j+1],  # diagonal neighbors
                laplacian[i-1, j+1], laplacian[i+1, j-1]
            ]
            
            center = laplacian[i, j]
            # If center has different sign than any neighbor, it's a zero crossing
            for neighbor in neighbors:
                if (center > 0 and neighbor < 0) or (center < 0 and neighbor > 0):
                    if abs(center - neighbor) > np.std(laplacian) * 0.5:  # Threshold for noise
                        zero_crossings[i, j] = 255
                        break
    
    return laplacian_result, zero_crossings.astype(np.uint8)

# Apply Laplacian edge detection with different sigma values
if cat_image is not None and dog_image is not None and hybrid_image_result is not None:
    
    # Apply Laplacian with different sigma values
    # Fine scale (small sigma - detects fine edges)
    cat_lap_fine, cat_zero_fine = laplacian_edge_detection(cat_image, "Cat", 0.8)
    dog_lap_fine, dog_zero_fine = laplacian_edge_detection(dog_image, "Dog", 0.8)
    hybrid_lap_fine, hybrid_zero_fine = laplacian_edge_detection(hybrid_image_result, "Hybrid", 0.8)
    
    # Medium scale (medium sigma - balanced)
    cat_lap_med, cat_zero_med = laplacian_edge_detection(cat_image, "Cat", 1.5)
    dog_lap_med, dog_zero_med = laplacian_edge_detection(dog_image, "Dog", 1.5)
    hybrid_lap_med, hybrid_zero_med = laplacian_edge_detection(hybrid_image_result, "Hybrid", 1.5)
    
    # Coarse scale (large sigma - detects major edges)
    cat_lap_coarse, cat_zero_coarse = laplacian_edge_detection(cat_image, "Cat", 2.5)
    dog_lap_coarse, dog_zero_coarse = laplacian_edge_detection(dog_image, "Dog", 2.5)
    hybrid_lap_coarse, hybrid_zero_coarse = laplacian_edge_detection(hybrid_image_result, "Hybrid", 2.5)
    
    # Display Laplacian magnitude results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat image results
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_lap_fine, cmap='gray')
    axes[0,1].set_title('Cat Laplacian Fine\n(σ=0.8)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_lap_med, cmap='gray')
    axes[0,2].set_title('Cat Laplacian Medium\n(σ=1.5)')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_lap_coarse, cmap='gray')
    axes[0,3].set_title('Cat Laplacian Coarse\n(σ=2.5)')
    axes[0,3].axis('off')
    
    # Row 2: Dog image results
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_lap_fine, cmap='gray')
    axes[1,1].set_title('Dog Laplacian Fine\n(σ=0.8)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_lap_med, cmap='gray')
    axes[1,2].set_title('Dog Laplacian Medium\n(σ=1.5)')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_lap_coarse, cmap='gray')
    axes[1,3].set_title('Dog Laplacian Coarse\n(σ=2.5)')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid image results
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_lap_fine, cmap='gray')
    axes[2,1].set_title('Hybrid Laplacian Fine\n(σ=0.8)')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_lap_med, cmap='gray')
    axes[2,2].set_title('Hybrid Laplacian Medium\n(σ=1.5)')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_lap_coarse, cmap='gray')
    axes[2,3].set_title('Hybrid Laplacian Coarse\n(σ=2.5)')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display Zero Crossings results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat zero crossings
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_zero_fine, cmap='gray')
    axes[0,1].set_title('Cat Zero Crossings Fine\n(σ=0.8)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_zero_med, cmap='gray')
    axes[0,2].set_title('Cat Zero Crossings Medium\n(σ=1.5)')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_zero_coarse, cmap='gray')
    axes[0,3].set_title('Cat Zero Crossings Coarse\n(σ=2.5)')
    axes[0,3].axis('off')
    
    # Row 2: Dog zero crossings
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_zero_fine, cmap='gray')
    axes[1,1].set_title('Dog Zero Crossings Fine\n(σ=0.8)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_zero_med, cmap='gray')
    axes[1,2].set_title('Dog Zero Crossings Medium\n(σ=1.5)')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_zero_coarse, cmap='gray')
    axes[1,3].set_title('Dog Zero Crossings Coarse\n(σ=2.5)')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid zero crossings
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_zero_fine, cmap='gray')
    axes[2,1].set_title('Hybrid Zero Crossings Fine\n(σ=0.8)')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_zero_med, cmap='gray')
    axes[2,2].set_title('Hybrid Zero Crossings Medium\n(σ=1.5)')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_zero_coarse, cmap='gray')
    axes[2,3].set_title('Hybrid Zero Crossings Coarse\n(σ=2.5)')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Laplacian edge detection completed!")
    print("Laplacian magnitude shows edge strength")
    print("Zero crossings show precise edge locations")
    print("Fine σ: Detects detailed edges and noise")
    print("Medium σ: Balanced edge detection")
    print("Coarse σ: Detects only major structural edges")

else:
    print("Error: Images not available for edge detection")