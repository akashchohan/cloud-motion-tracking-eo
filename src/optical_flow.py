import cv2
import numpy as np
from load_images import load_images

img1, img2 = load_images(
    "data/sample_images/cloud_1.png",
    "data/sample_images/cloud_2.png"
)

flow = cv2.calcOpticalFlowFarneback(
    img1,
    img2,
    None,
    0.5,
    3,
    15,
    3,
    5,
    1.2,
    0
)

magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

print("Average cloud motion magnitude:", np.mean(magnitude))