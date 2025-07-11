import cv2
import numpy as np
import os
import glob



def get_image_files(folder_path):
    image_files = []
    for file in glob.glob(os.path.join(folder_path, "*.png")):
        image_files.append(file)
    return image_files


def average_stack_with_rejection(images):
    # Create an empty pixel stack
    pixel_stack = np.zeros_like(images[0], dtype=np.float32)

    # Add pixel values of each image to the stack
    for image in images:
        pixel_stack += image.astype(np.float32)

    # Perform rejection by excluding outliers
    num_images = len(images)
    pixel_stack /= num_images

    # Convert the pixel stack back to uint8
    stacked_image = pixel_stack.astype(np.uint8)

    return stacked_image

# Load the images
folder='images folder'

#image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
image_files = get_image_files(folder)

print("Images files are: "+str(image_files))
images = [cv2.imread(path) for path in image_files]

# Perform average stacking with rejection
stacked_image = average_stack_with_rejection(images)


cv2.imwrite(folder2+'stacked_image_exp80_gain40_temp357_365.png', stacked_image)


# Display the stacked image
cv2.imshow("Stacked Image", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()