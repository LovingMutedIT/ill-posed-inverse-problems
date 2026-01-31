import numpy as np
import cv2
import matplotlib.pyplot as plt

mixed1 = cv2.imread('mixed2-1.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
mixed2 = cv2.imread('mixed2-2.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)


shape = mixed1.shape


x1 = mixed1.flatten()
x2 = mixed2.flatten()


X = np.vstack((x1, x2)) # Shape (2, N)
print("Data matrix shape:", X.shape)

X_mean = X.mean(axis=1, keepdims=True)
X_centered = X - X_mean

plt.figure()
plt.hist(X_centered[0], bins=100, density=True)
plt.title("Histogram of Centered Mixed Signal 1")
plt.show()

plt.figure()
plt.hist(X_centered[1], bins=100, density=True)
plt.title("Histogram of Centered Mixed Signal 2")
plt.show()

cov = np.cov(X_centered)
eigvals, eigvecs = np.linalg.eigh(cov)

D_inv = np.diag(1.0 / np.sqrt(eigvals))
whitening_matrix = D_inv @ eigvecs.T
X_white = whitening_matrix @ X_centered

# Visual proof: covariance should be identity
print("Covariance after whitening:\n", np.cov(X_white))

plt.figure()
plt.scatter(X_centered[0][:5000], X_centered[1][:5000], s=1)
plt.title("Before Whitening")
plt.show()

plt.figure()
plt.scatter(X_white[0][:5000], X_white[1][:5000], s=1)
plt.title("After Whitening (Spherical Data)")
plt.show()

def g(u):
    return np.tanh(u)

def g_prime(u):
    return 1 - np.tanh(u)**2

W = np.random.randn(2, 2)

for i in range(1000):
    W_old = W.copy()
    
    WX = W @ X_white
    W = (g(WX) @ X_white.T) / X_white.shape[1] - np.diag(g_prime(WX).mean(axis=1)) @ W
    
    # Orthogonalize W
    U, _, Vt = np.linalg.svd(W)
    W = U @ Vt

    # Convergence check
    if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < 1e-6:
        break

print("FastICA converged in", i, "iterations")

S_est = W @ X_white

s1_est = S_est[0].reshape(shape)
s2_est = S_est[1].reshape(shape)

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img

s1_disp = normalize(s1_est)
s2_disp = normalize(s2_est)
m1_disp = normalize(mixed1)
m2_disp = normalize(mixed2)

plt.figure()
plt.imshow(s1_disp, cmap='gray')
plt.title("Recovered Source 1")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(s2_disp, cmap='gray')
plt.title("Recovered Source 2")
plt.axis('off')
plt.show()

top = np.hstack((m1_disp, m2_disp))
bottom = np.hstack((s1_disp, s2_disp))
comparison = np.vstack((top, bottom))

plt.figure(figsize=(8,8))
plt.imshow(comparison, cmap='gray')
plt.title("Top: Mixed Images | Bottom: ICA Recovered Sources")
plt.axis('off')
plt.show()

cv2.imwrite("ica_comparison_2.png", (comparison*255).astype(np.uint8))