import cv2
import numpy as np


def create_test_image(width=512, height=512):
    """Create a test image with various patterns for VRS testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient background
    for y in range(height):
        for x in range(width):
            image[y, x] = [
                int(255 * x / width),  # Blue gradient
                int(255 * y / height),  # Green gradient
                128  # Constant red
            ]

    # Add some bright spots (to test content-aware luminance)
    cv2.circle(image, (100, 100), 20, (255, 255, 255), -1)
    cv2.circle(image, (400, 100), 15, (255, 255, 0), -1)
    cv2.circle(image, (250, 250), 30, (0, 255, 255), -1)
    cv2.circle(image, (100, 400), 25, (255, 0, 255), -1)
    cv2.circle(image, (400, 400), 18, (200, 200, 255), -1)

    # Add some rectangles
    cv2.rectangle(image, (50, 200), (150, 300), (255, 128, 0), -1)
    cv2.rectangle(image, (350, 200), (450, 300), (0, 128, 255), -1)

    # Add some lines for high-frequency detail
    for i in range(0, width, 10):
        cv2.line(image, (i, 350), (i, 380), (255, 255, 255), 1)

    for i in range(0, height, 10):
        cv2.line(image, (150, i), (180, i), (0, 0, 0), 1)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "VRS TEST", (200, 50), font, 1, (255, 255, 255), 2)

    return image


if __name__ == "__main__":
    # Generate test image
    test_image = create_test_image()

    # Save test image
    cv2.imwrite("test_input.png", test_image)
    print("Test image saved as test_input.png")