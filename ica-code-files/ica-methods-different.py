import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# -------------------------------------------------
# Load Mixed Images
# -------------------------------------------------
mixed1 = cv2.imread('mixed1.png', 0).astype(np.float64)
mixed2 = cv2.imread('mixed2.png', 0).astype(np.float64)
shape = mixed1.shape

X = np.vstack([mixed1.flatten(), mixed2.flatten()])

# -------------------------------------------------
# Centering
# -------------------------------------------------
X -= X.mean(axis=1, keepdims=True)

# -------------------------------------------------
# Whitening
# -------------------------------------------------
cov = np.cov(X)
eigvals, eigvecs = np.linalg.eigh(cov)
D_inv = np.diag(1.0 / np.sqrt(eigvals))
whitening = D_inv @ eigvecs.T
X_white = whitening @ X
N = X_white.shape[1]

# Utility
def normalize(img):
    img -= img.min()
    img /= img.max()
    return img

# =================================================
# 1️⃣ INFOMAX ICA
# =================================================
start = time.time()
W = np.eye(2)

for _ in range(500):
    Y = W @ X_white
    g = np.tanh(Y)
    dW = np.eye(2) + (1 - 2*g) @ Y.T / N
    W += 0.01 * dW @ W

S_infomax = W @ X_white
t_infomax = time.time() - start

""" # =================================================
# 2️⃣ JADE (Simplified Cumulant Diagonalization)
# =================================================
start = time.time()

def jade_2d(X):
    X1, X2 = X
    Q = np.array([
        [np.mean(X1**4)-3, np.mean(X1**3 * X2)],
        [np.mean(X1 * X2**3), np.mean(X2**4)-3]
    ])
    U, _, Vt = np.linalg.svd(Q)
    return U @ Vt

W_jade = jade_2d(X_white)
S_jade = W_jade @ X_white
t_jade = time.time() - start
 """

# =================================================
# 2️⃣ JADE (True Implementation)
# =================================================
start = time.time()

k = X_white.shape[0]

# ----------- 4th Order Cumulant Matrices -----------
def cumulant_matrix(X, i, j):
    xi = X[i]
    xj = X[j]
    C = np.zeros((k, k))
    for p in range(k):
        xp = X[p]
        for q in range(k):
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

# ----------- Joint Diagonalization -----------
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
                    G[p,p] = c
                    G[q,q] = c
                    G[p,q] = -s
                    G[q,p] = s

                    for idx in range(len(matrices)):
                        matrices[idx] = G.T @ matrices[idx] @ G
                    V = V @ G
        if off < eps:
            break
    return V

V = joint_diagonalize(cumulant_matrices)

# JADE separation matrix (undo whitening)
W_jade = V.T @ whitening
S_jade = W_jade @ X

t_jade = time.time() - start


# =================================================
# 3️⃣ MAXIMUM LIKELIHOOD ICA
# =================================================
start = time.time()
W = np.random.randn(2,2)

for _ in range(500):
    Y = W @ X_white
    g = np.tanh(Y)
    dW = (np.eye(2) - g @ Y.T / N) @ W
    W += 0.01 * dW

S_ml = W @ X_white
t_ml = time.time() - start

# =================================================
# 4️⃣ FASTICA (Reference)
# =================================================
start = time.time()
W = np.random.randn(2,2)

for _ in range(500):
    WX = W @ X_white
    W = (np.tanh(WX) @ X_white.T)/N - np.diag((1-np.tanh(WX)**2).mean(axis=1)) @ W
    U, _, Vt = np.linalg.svd(W)
    W = U @ Vt

S_fast = W @ X_white
t_fast = time.time() - start

# -------------------------------------------------
# Reshape Results
# -------------------------------------------------
def reshape_sources(S):
    return normalize(S[0].reshape(shape)), normalize(S[1].reshape(shape))

inf1, inf2 = reshape_sources(S_infomax)
jade1, jade2 = reshape_sources(S_jade)
ml1, ml2 = reshape_sources(S_ml)
fast1, fast2 = reshape_sources(S_fast)

# -------------------------------------------------
# Plot Comparison
# -------------------------------------------------
fig, axs = plt.subplots(4, 2, figsize=(8,12))
axs[0,0].imshow(inf1, cmap='gray'); axs[0,0].set_title("Infomax 1")
axs[0,1].imshow(inf2, cmap='gray'); axs[0,1].set_title("Infomax 2")

axs[1,0].imshow(jade1, cmap='gray'); axs[1,0].set_title("JADE 1")
axs[1,1].imshow(jade2, cmap='gray'); axs[1,1].set_title("JADE 2")

axs[2,0].imshow(ml1, cmap='gray'); axs[2,0].set_title("ML ICA 1")
axs[2,1].imshow(ml2, cmap='gray'); axs[2,1].set_title("ML ICA 2")

axs[3,0].imshow(fast1, cmap='gray'); axs[3,0].set_title("FastICA 1")
axs[3,1].imshow(fast2, cmap='gray'); axs[3,1].set_title("FastICA 2")

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Time Complexity Plot
# -------------------------------------------------
times = [t_infomax, t_jade, t_ml, t_fast]
labels = ['Infomax', 'JADE', 'Max Likelihood', 'FastICA']

plt.figure()
plt.bar(labels, times)
plt.ylabel("Execution Time (seconds)")
plt.title("ICA Algorithm Time Comparison")
plt.show()

print("Execution Times:")
print("Infomax:", t_infomax)
print("JADE:", t_jade)
print("Max Likelihood:", t_ml)
print("FastICA:", t_fast)