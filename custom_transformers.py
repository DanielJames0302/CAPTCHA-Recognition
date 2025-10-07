import cv2
import numpy as np
from mltu.annotations.images import CVImage


class GrayscaleTransformer:
    """Custom transformer that converts RGB images to grayscale"""
    
    def __init__(self, keep_3_channels=True):
        """
        Args:
            keep_3_channels (bool): If True, converts to grayscale but keeps 3 channels
                                  If False, outputs single channel grayscale
        """
        self.keep_3_channels = keep_3_channels
    
    def __call__(self, image, label):
        """
        Convert image to grayscale and return with label unchanged.
        
        Args:
            image: Input image (CVImage or numpy array)
            label: Text label (unchanged)
        
        Returns:
            Tuple of (grayscale_image, label)
        """
        # Handle CVImage objects
        if hasattr(image, 'numpy'):
            image_array = image.numpy()
        else:
            image_array = image
            
        # Convert to grayscale
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Convert RGB to grayscale using OpenCV
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            if self.keep_3_channels:
                # Convert back to 3-channel format
                grayscale_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                # Keep as single channel, add channel dimension
                grayscale_image = np.expand_dims(gray, axis=-1)
        else:
            # Already grayscale or wrong format
            grayscale_image = image_array
        
        # Convert back to CVImage if input was CVImage
        if hasattr(image, 'numpy'):
            result_image = CVImage(grayscale_image)
        else:
            result_image = grayscale_image
            
        return result_image, label

def test_grayscale():
    """Test the GrayscaleTransformer"""
    import numpy as np
    
    # Create a test RGB image
    test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    test_label = "test123"
    
    # Test with 3 channels output
    transformer_3ch = GrayscaleTransformer(keep_3_channels=True)
    result_image_3ch, result_label = transformer_3ch(test_image, test_label)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Grayscale (3ch) output shape: {result_image_3ch.shape}")
    print(f"Label unchanged: {result_label}")
    
    # Test with 1 channel output
    transformer_1ch = GrayscaleTransformer(keep_3_channels=False)
    result_image_1ch, result_label = transformer_1ch(test_image, test_label)
    
    print(f"Grayscale (1ch) output shape: {result_image_1ch.shape}")
    
    # Verify shapes
    assert result_image_3ch.shape == (100, 200, 3), "Wrong 3-channel output shape!"
    assert result_image_1ch.shape == (100, 200, 1), "Wrong 1-channel output shape!"
    assert result_label == test_label, "Label was modified!"
    
    print("Grayscale transformer test passed!")


if __name__ == "__main__":
    test_grayscale()