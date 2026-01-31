import numpy as np
import cv2


# Load images
img1 = cv2.imread('mona-lisa.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2 = cv2.imread('sample-watermark.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)


# Resize if needed
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


# Zero-mean
img1 -= np.mean(img1)
img2 -= np.mean(img2)

s1 = img1.flatten()
s2 = img2.flatten()

S = np.vstack((s1, s2))   # Shape: (2, N)

# declaring the mixing matrix directly
A = np.array([[1.0, 0.005],
              [0.003, 1.0]], dtype=np.float32)

X = A @ S   # Matrix multiplication

mixed1 = X[0].reshape(img1.shape)
mixed2 = X[1].reshape(img1.shape)


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return (img * 255).astype(np.uint8)

cv2.imwrite("mixed2-1.png", normalize(mixed1))
cv2.imwrite("mixed2-2.png", normalize(mixed2))