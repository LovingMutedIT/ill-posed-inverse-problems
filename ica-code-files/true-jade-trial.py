import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load Mixed Images
# -------------------------------------------------
mixed1 = cv2.imread('mixed1.png', 0).astype(np.float64)
mixed2 = cv2.imread('mixed2.png', 0).astype(np.float64)
shape = mixed1.shape

X = np.vstack([mixed1.flatten(), mixed2.flatten()])
N = X.shape[1]

# -------------------------------------------------
# 1. Center
# -------------------------------------------------
X -= X.mean(axis=1, keepdims=True)

# -------------------------------------------------
# 2. Whiten
# -------------------------------------------------
cov = np.cov(X)
eigvals, eigvecs = np.linalg.eigh(cov)
D_inv = np.diag(1.0 / np.sqrt(eigvals))
whitening = D_inv @ eigvecs.T
X_white = whitening @ X
k = X_white.shape[0]

# -------------------------------------------------
# 3. Compute 4th Order Cumulant Matrices
# -------------------------------------------------
def cumulant_matrix(X, i, j):
    xi = X[i]
    xj = X[j]
    term1 = np.mean((xi**2) * (xj**2))
    term2 = np.mean(xi**2) * np.mean(xj**2)
    term3 = 2 * (np.mean(xi * xj))**2
    val = term1 - term2 - term3

    C = np.zeros((k, k))
    for p in range(k):
        for q in range(k):
            xp = X[p]
            xq = X[q]
            C[p, q] = np.mean(xi * xj * xp * xq) \
                      - np.mean(xi * xj) * np.mean(xp * xq) \
                      - np.mean(xi * xp) * np.mean(xj * xq) \
                      - np.mean(xi * xq) * np.mean(xj * xp)
    return C

cumulant_matrices = []
for i in range(k):
    for j in range(i, k):
        cumulant_matrices.append(cumulant_matrix(X_white, i, j))

# -------------------------------------------------
# 4. Joint Diagonalization (Jacobi Rotations)
# -------------------------------------------------
def joint_diagonalize(matrices, eps=1e-6, max_iter=100):
    V = np.eye(k)
    for _ in range(max_iter):
        off = 0
        for p in range(k-1):
            for q in range(p+1, k):
                g11 = g22 = g12 = g21 = 0
                for M in matrices:
                    g11 += M[p,p] - M[q,q]
                    g12 += M[p,q] + M[q,p]
                    g21 += M[p,q] - M[q,p]
                    g22 += M[p,p] + M[q,q]

                theta = 0.5 * np.arctan2(2*g12, g11 - g22 + 1e-12)
                c = np.cos(theta)
                s = np.sin(theta)

                if abs(s) > eps:
                    off += abs(s)
                    G = np.eye(k)
                    G[[p,p,q,q],[p,q,p,q]] = c, -s, s, c

                    for i in range(len(matrices)):
                        matrices[i] = G.T @ matrices[i] @ G
                    V = V @ G
        if off < eps:
            break
    return V

V = joint_diagonalize(cumulant_matrices)

# -------------------------------------------------
# 5. Recover Sources
# -------------------------------------------------
W_jade = V.T @ whitening
S_jade = W_jade @ X

s1 = S_jade[0].reshape(shape)
s2 = S_jade[1].reshape(shape)

# Normalize for display
def normalize(img):
    img -= img.min()
    img /= img.max()
    return img

s1 = normalize(s1)
s2 = normalize(s2)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(s1, cmap='gray')
plt.title("JADE Recovered Source 1")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(s2, cmap='gray')
plt.title("JADE Recovered Source 2")
plt.axis('off')
plt.show()