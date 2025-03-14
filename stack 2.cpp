import cv2
import numpy as np

def lbp(image):
  """
  Calculates the Local Binary Pattern (LBP) representation of an image.

  Args:
    image: The input image (grayscale or color).

  Returns:
    The LBP image (3-channel).
  """

  # Convert the image to grayscale if it's not already
  if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray = image

  # Calculate LBP for each pixel
  lbp_image = np.zeros_like(gray)
  for i in range(1, gray.shape[0] - 1):
    for j in range(1, gray.shape[1] - 1):
      center = gray[i, j]
      code = 0
      code |= (gray[i - 1, j - 1] > center) << 7
      code |= (gray[i - 1, j] > center) << 6
      code |= (gray[i - 1, j + 1] > center) << 5
      code |= (gray[i, j + 1] > center) << 4
      code |= (gray[i + 1, j + 1] > center) << 3
      code |= (gray[i + 1, j] > center) << 2
      code |= (gray[i + 1, j - 1] > center) << 1
      code |= (gray[i, j - 1] > center) << 0
      lbp_image[i, j] = code

  # Convert the LBP image to 3 channels
  lbp_image = cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR)

  return lbp_image
