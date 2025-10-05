import cv2
import numpy as np
from mltu.annotations.images import CVImage

class AspectRatioImageResizer:
    """Custom transformer for mltu that maintains aspect ratio with padding"""
    
    def __init__(self, width, height, pad_color=(0, 0, 0), left_align=True):
        self.width = width
        self.height = height
        self.pad_color = pad_color
        self.left_align = left_align
    
    def __call__(self, image, label):
        """
        Process image and return resized version with label unchanged.
        
        Args:
            image: Input image (numpy array)
            label: Text label (unchanged)
        
        Returns:
            Tuple of (processed_image, label)
        """
        processed_image = self.resize_with_padding(image)
        return processed_image, label
    
    def resize_with_padding(self, image):
        # Convert CVImage to numpy array
        image = image.numpy()

        """Resize while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if self.left_align:
            # Scale to fit height, then pad width
            scale = self.height / h
            new_w = int(w * scale)
            new_h = self.height
            
            # If too wide, scale to fit width
            if new_w > self.width:
                scale = self.width / w
                new_w = self.width
                new_h = int(h * scale)
        else:
            # Scale to fit within both dimensions
            scale = min(self.width / w, self.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.full((self.height, self.width, 3), self.pad_color, dtype=np.uint8)
        
        if self.left_align:
            # Left-align for text
            y_offset = (self.height - new_h) // 2
            padded[y_offset:y_offset+new_h, 0:new_w] = resized
        else:
            # Center the image
            y_offset = (self.height - new_h) // 2
            x_offset = (self.width - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded


def test_resizer():
    """Simple test to verify AspectRatioImageResizer works"""
    import numpy as np
    
    # Create a test image (100x200 pixels)
    test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    # Test the resizer
    resizer = AspectRatioImageResizer(400, 80, left_align=True)
    result_image, result_label = resizer(test_image, "")
    
    # Verify output
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result_image.shape}")
    
    assert result_image.shape == (80, 400, 3), "Wrong output shape!"
    print("Test passed!")


if __name__ == "__main__":
    test_resizer()