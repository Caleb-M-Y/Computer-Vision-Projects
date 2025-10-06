# Canny Edge Detection Method
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from Project_2_Hybrid_Img import cat_image, dog_image, hybrid_image_result

def canny_edge_detection(img, image_name, low_threshold=50, high_threshold=150, sigma=1.0):
    """
    Apply Canny edge detection to an image
    
    Parameters:
    - img: Input image (color or grayscale)
    - image_name: Name for display purposes
    - low_threshold: Lower threshold for edge linking
    - high_threshold: Upper threshold for edge detection
    - sigma: Standard deviation for Gaussian blur
    
    Returns:
    - canny_edges: Canny edge detection result
    """
    
    # Convert to grayscale for edge detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Convert to uint8 if it's not already (this fixes the hybrid image issue)
    if gray.dtype != np.uint8:
        # If it's float in range [0,1], scale to [0,255]
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            # If it's float in range [0,255], just convert
            gray = gray.astype(np.uint8)

    # Apply Gaussian blur with specified sigma
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return canny_edges

# Apply Canny edge detection with different parameters
if cat_image is not None and dog_image is not None and hybrid_image_result is not None:
    
    # Apply Canny with different threshold values to show sensitivity
    # Standard parameters
    cat_canny_std = canny_edge_detection(cat_image, "Cat", 50, 150, 1.0)
    dog_canny_std = canny_edge_detection(dog_image, "Dog", 50, 150, 1.0)
    hybrid_canny_std = canny_edge_detection(hybrid_image_result, "Hybrid", 50, 150, 1.0)
    
    # Sensitive parameters (lower thresholds - more edges)
    cat_canny_sensitive = canny_edge_detection(cat_image, "Cat", 20, 80, 0.8)
    dog_canny_sensitive = canny_edge_detection(dog_image, "Dog", 20, 80, 0.8)
    hybrid_canny_sensitive = canny_edge_detection(hybrid_image_result, "Hybrid", 20, 80, 0.8)
    
    # Conservative parameters (higher thresholds - fewer edges)
    cat_canny_conservative = canny_edge_detection(cat_image, "Cat", 100, 200, 1.5)
    dog_canny_conservative = canny_edge_detection(dog_image, "Dog", 100, 200, 1.5)
    hybrid_canny_conservative = canny_edge_detection(hybrid_image_result, "Hybrid", 100, 200, 1.5)
    
    # Display results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat image results
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_canny_sensitive, cmap='gray')
    axes[0,1].set_title('Cat Canny Sensitive\n(Low=20, High=80)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_canny_std, cmap='gray')
    axes[0,2].set_title('Cat Canny Standard\n(Low=50, High=150)')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_canny_conservative, cmap='gray')
    axes[0,3].set_title('Cat Canny Conservative\n(Low=100, High=200)')
    axes[0,3].axis('off')
    
    # Row 2: Dog image results
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_canny_sensitive, cmap='gray')
    axes[1,1].set_title('Dog Canny Sensitive\n(Low=20, High=80)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_canny_std, cmap='gray')
    axes[1,2].set_title('Dog Canny Standard\n(Low=50, High=150)')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_canny_conservative, cmap='gray')
    axes[1,3].set_title('Dog Canny Conservative\n(Low=100, High=200)')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid image results
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_canny_sensitive, cmap='gray')
    axes[2,1].set_title('Hybrid Canny Sensitive\n(Low=20, High=80)')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_canny_std, cmap='gray')
    axes[2,2].set_title('Hybrid Canny Standard\n(Low=50, High=150)')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_canny_conservative, cmap='gray')
    axes[2,3].set_title('Hybrid Canny Conservative\n(Low=100, High=200)')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Canny edge detection completed!")
    print("Sensitive: Detects more edges (lower thresholds)")
    print("Standard: Balanced edge detection") 
    print("Conservative: Detects only strong edges (higher thresholds)")
    print("Canny uses hysteresis thresholding for clean, connected edges")

else:
    print("Error: Images not available for edge detection")