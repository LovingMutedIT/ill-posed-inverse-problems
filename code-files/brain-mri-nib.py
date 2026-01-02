import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

img = nib.load("../datasets/test_subject_1.nii.gz")
data = img.get_fdata()
slice_ = data[:, :, data.shape[2] // 2]
slice_ = slice_ / slice_.max()
slice_ = resize(slice_, (128, 128), anti_aliasing=True)
n, m = slice_.shape


# simulating motion blur + sensor noise
def gaussian_blur_1d(n, sigma):
    x = np.arange(n)
    A = np.zeros((n, n))
    for i in range(n): 
        A[i] = np.exp(-(x-i)**2 / (2*sigma**2))
    A /= A.sum(axis=1, keepdims=True)
    return A

Ax = gaussian_blur_1d(n, sigma=2.5)
Ay = gaussian_blur_1d(m, sigma=2.5)

B = Ax @ slice_ @Ay.T
B += 0.02 * np.random.randn(*B.shape)

def tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, B, k):
    Bk = Ux[:, :k].T @ B @ Uy[:, :k]

    Bk /= np.outer(Sx[:k], Sy[:k])

    return VxT[:k, :].T @ Bk @ VyT[:k, :]

def tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha):
    B_hat = Ux.T @ B @ Uy
    X_hat = np.zeros_like(B_hat)

    lam2 = np.outer(Sx**2, Sy**2)
    X_hat = (lam2 / (lam2 + alpha)) * B_hat / np.sqrt(lam2)

    return VxT.T @ X_hat @ VyT

Ux, Sx, VxT = np.linalg.svd(Ax, full_matrices=False)
Uy, Sy, VyT = np.linalg.svd(Ay, full_matrices=False)

k = int(0.1 * n)


x_tsvd = tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, B, k)
# choosing the best value for alpha
delta = np.sqrt(n * m) * 0.02
errors = []
alphas = np.logspace(-6, 0, 30)
for alpha in alphas:
    X = tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha)
    residual = Ax @ X @ Ay.T - B
    errors.append(abs(np.linalg.norm(residual) - delta))

alpha_opt = alphas[np.argmin(errors)]

x_tikhonov = tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha_opt)


from skimage.filters import sobel

edge_true = sobel(slice_)
edge_tsvd = sobel(x_tsvd)
edge_tikh = sobel(x_tikhonov)


plt.figure(figsize=(12,8))

plt.subplot(3, 3, 1)
plt.imshow(slice_, cmap='gray')
plt.title("Original MRI")

plt.subplot(3, 3, 2)
plt.imshow(B, cmap='gray')
plt.title("Degraded")

plt.subplot(3, 3, 3)
plt.imshow(x_tsvd, cmap='gray')
plt.title("TSVD")

plt.subplot(3,3,4)
plt.imshow(edge_true, cmap='gray')
plt.title("True edges")

plt.subplot(3,3,5)
plt.imshow(edge_tsvd, cmap='gray')
plt.title("TSVD edges")

plt.subplot(3,3,6)
plt.imshow(edge_tikh, cmap='gray')
plt.title("Tikhonov edges")

plt.subplot(3,3,7)
plt.imshow(x_tikhonov, cmap='gray')
plt.title("Tikhonov")

plt.subplot(3,3,8)
plt.axis("off")
    

plt.tight_layout()
plt.show()
