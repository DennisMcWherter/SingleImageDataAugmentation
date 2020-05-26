import cv2

def load_rgb_image(path, size=(224,224)):
    # Load an image in RGB format with channels-last (i.e. (H, W, 3))
    # Pixel values range from [0, 255]
    return cv2.cvtColor(cv2.resize(cv2.imread(path), size), cv2.COLOR_BGR2RGB)

