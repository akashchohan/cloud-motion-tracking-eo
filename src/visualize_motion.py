import cv2
import numpy as np
import matplotlib.pyplot as plt
from load_images import load_images

img1, img2 = load_images(
    "data/sample_images/cloud_1.png",
    "data/sample_images/cloud_2.png"
)

flow = cv2.calcOpticalFlowFarneback(img1, img2, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0)

step = 16
h, w = img1.shape
y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
fx, fy = flow[y.astype(int), x.astype(int)].T

plt.imshow(img1, cmap='gray')
plt.quiver(x, y, fx, fy, color='red')
plt.title("Cloud Motion Vectors")
plt.show()