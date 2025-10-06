# We will use the resulting hybrid image from the 
# image above to do edge detection as well as the 
# cat and dog images. 

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from Project_2_Hybrid_Img import cat_image, dog_image, hybrid_image_result

# Sobel Edge Detection Method

def sobel_edge_detection(img, image_name, ksize=3):
    """
    Apply Sobel edge detection to an image
    
    Parameters:
    - img: Input image (color or grayscale)
    - image_name: Name for display purposes
    - ksize: Size of the Sobel kernel (1, 3, 5, or 7)
    
    Returns:
    - sobel_x: Sobel X gradient (vertical edges)
    - sobel_y: Sobel Y gradient (horizontal edges)
    - sobel_combined: Combined Sobel magnitude
    """
    
    # Convert to grayscale for edge detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Apply Sobel operator in X and Y directions
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)  # Vertical edges
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)  # Horizontal edges
    
    # Calculate magnitude (combined gradients)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Convert to uint8 for display
    sobel_x_display = np.uint8(np.absolute(sobel_x))
    sobel_y_display = np.uint8(np.absolute(sobel_y))
    sobel_combined_display = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    return sobel_x_display, sobel_y_display, sobel_combined_display

# Apply Sobel edge detection with different kernel sizes
if cat_image is not None and dog_image is not None and hybrid_image_result is not None:
    
    # Apply Sobel with different kernel sizes
    # Small kernel (3x3 - detects fine edges)
    cat_sobel_x_3, cat_sobel_y_3, cat_sobel_combined_3 = sobel_edge_detection(cat_image, "Cat", 3)
    dog_sobel_x_3, dog_sobel_y_3, dog_sobel_combined_3 = sobel_edge_detection(dog_image, "Dog", 3)
    hybrid_sobel_x_3, hybrid_sobel_y_3, hybrid_sobel_combined_3 = sobel_edge_detection(hybrid_image_result, "Hybrid", 3)
    
    # Medium kernel (5x5 - balanced)
    cat_sobel_x_5, cat_sobel_y_5, cat_sobel_combined_5 = sobel_edge_detection(cat_image, "Cat", 5)
    dog_sobel_x_5, dog_sobel_y_5, dog_sobel_combined_5 = sobel_edge_detection(dog_image, "Dog", 5)
    hybrid_sobel_x_5, hybrid_sobel_y_5, hybrid_sobel_combined_5 = sobel_edge_detection(hybrid_image_result, "Hybrid", 5)
    
    # Large kernel (7x7 - detects major edges)
    cat_sobel_x_7, cat_sobel_y_7, cat_sobel_combined_7 = sobel_edge_detection(cat_image, "Cat", 7)
    dog_sobel_x_7, dog_sobel_y_7, dog_sobel_combined_7 = sobel_edge_detection(dog_image, "Dog", 7)
    hybrid_sobel_x_7, hybrid_sobel_y_7, hybrid_sobel_combined_7 = sobel_edge_detection(hybrid_image_result, "Hybrid", 7)
    
    # Display X gradients (vertical edges)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat X gradients
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_sobel_x_3, cmap='gray')
    axes[0,1].set_title('Cat Sobel X (3x3)\nVertical Edges')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_sobel_x_5, cmap='gray')
    axes[0,2].set_title('Cat Sobel X (5x5)\nVertical Edges')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_sobel_x_7, cmap='gray')
    axes[0,3].set_title('Cat Sobel X (7x7)\nVertical Edges')
    axes[0,3].axis('off')
    
    # Row 2: Dog X gradients
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_sobel_x_3, cmap='gray')
    axes[1,1].set_title('Dog Sobel X (3x3)\nVertical Edges')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_sobel_x_5, cmap='gray')
    axes[1,2].set_title('Dog Sobel X (5x5)\nVertical Edges')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_sobel_x_7, cmap='gray')
    axes[1,3].set_title('Dog Sobel X (7x7)\nVertical Edges')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid X gradients
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_sobel_x_3, cmap='gray')
    axes[2,1].set_title('Hybrid Sobel X (3x3)\nVertical Edges')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_sobel_x_5, cmap='gray')
    axes[2,2].set_title('Hybrid Sobel X (5x5)\nVertical Edges')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_sobel_x_7, cmap='gray')
    axes[2,3].set_title('Hybrid Sobel X (7x7)\nVertical Edges')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display Y gradients (horizontal edges)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat Y gradients
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_sobel_y_3, cmap='gray')
    axes[0,1].set_title('Cat Sobel Y (3x3)\nHorizontal Edges')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_sobel_y_5, cmap='gray')
    axes[0,2].set_title('Cat Sobel Y (5x5)\nHorizontal Edges')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_sobel_y_7, cmap='gray')
    axes[0,3].set_title('Cat Sobel Y (7x7)\nHorizontal Edges')
    axes[0,3].axis('off')
    
    # Row 2: Dog Y gradients
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_sobel_y_3, cmap='gray')
    axes[1,1].set_title('Dog Sobel Y (3x3)\nHorizontal Edges')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_sobel_y_5, cmap='gray')
    axes[1,2].set_title('Dog Sobel Y (5x5)\nHorizontal Edges')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_sobel_y_7, cmap='gray')
    axes[1,3].set_title('Dog Sobel Y (7x7)\nHorizontal Edges')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid Y gradients
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_sobel_y_3, cmap='gray')
    axes[2,1].set_title('Hybrid Sobel Y (3x3)\nHorizontal Edges')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_sobel_y_5, cmap='gray')
    axes[2,2].set_title('Hybrid Sobel Y (5x5)\nHorizontal Edges')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_sobel_y_7, cmap='gray')
    axes[2,3].set_title('Hybrid Sobel Y (7x7)\nHorizontal Edges')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display combined magnitude
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Cat combined
    axes[0,0].imshow(cat_image)
    axes[0,0].set_title('Original Cat')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cat_sobel_combined_3, cmap='gray')
    axes[0,1].set_title('Cat Sobel Combined (3x3)\nAll Edges')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(cat_sobel_combined_5, cmap='gray')
    axes[0,2].set_title('Cat Sobel Combined (5x5)\nAll Edges')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cat_sobel_combined_7, cmap='gray')
    axes[0,3].set_title('Cat Sobel Combined (7x7)\nAll Edges')
    axes[0,3].axis('off')
    
    # Row 2: Dog combined
    axes[1,0].imshow(dog_image)
    axes[1,0].set_title('Original Dog')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dog_sobel_combined_3, cmap='gray')
    axes[1,1].set_title('Dog Sobel Combined (3x3)\nAll Edges')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(dog_sobel_combined_5, cmap='gray')
    axes[1,2].set_title('Dog Sobel Combined (5x5)\nAll Edges')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(dog_sobel_combined_7, cmap='gray')
    axes[1,3].set_title('Dog Sobel Combined (7x7)\nAll Edges')
    axes[1,3].axis('off')
    
    # Row 3: Hybrid combined
    axes[2,0].imshow(hybrid_image_result)
    axes[2,0].set_title('Original Hybrid')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(hybrid_sobel_combined_3, cmap='gray')
    axes[2,1].set_title('Hybrid Sobel Combined (3x3)\nAll Edges')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(hybrid_sobel_combined_5, cmap='gray')
    axes[2,2].set_title('Hybrid Sobel Combined (5x5)\nAll Edges')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(hybrid_sobel_combined_7, cmap='gray')
    axes[2,3].set_title('Hybrid Sobel Combined (7x7)\nAll Edges')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Sobel edge detection completed!")
    print("X gradients: Detect vertical edges")
    print("Y gradients: Detect horizontal edges") 
    print("Combined: Shows all edge directions")
    print("3x3: Fine detail edges")
    print("5x5: Balanced edge detection")
    print("7x7: Major structural edges")

else:
    print("Error: Images not available for edge detection")