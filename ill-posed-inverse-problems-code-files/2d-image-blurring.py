import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from skimage import data, color, img_as_float
from scipy.linalg import svd
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# loading ground truth image 
img = img_as_float(data.camera())
img = img[::4, ::4]
n, m  = img.shape

plt.imshow(img, cmap='gray')
plt.title("Ground Truth Image")
plt.axis("off")


# constructing the 2d gaussian blur matrix using 1D and seperatability
def gaussian_blur_1d(n, sigma):
    x = np.arange(n)
    A = np.zeros((n, n))
    for i in range(n): 
        A[i] = np.exp(-(x-i)**2 / (2*sigma**2))
    A /= A.sum(axis=1, keepdims=True)
    return A

#using this 1d blur, we make the 2d blur matrix
sigma = 2.0
Ax = gaussian_blur_1d(n, sigma)
Ay = gaussian_blur_1d(m, sigma)


noise_level = 0.01
blurred = Ax @ img @ Ay.T
blurred += noise_level * np.random.randn(*blurred.shape)

plt.imshow(blurred, cmap='gray')
plt.title("Blurred + Noisy observation")
plt.axis("off")


# now we are calculating svd of 1d operators individually. this will become mathematically equivalent to SVD of the full 2D blur
Ux, Sx, VxT = svd(Ax, full_matrices=False)
Uy, Sy, VyT = svd(Ay, full_matrices=False)

""" def tsvd_2d(Ux, Sx, VxT, Uy, Sy, VyT, B, k):
    B_hat = Ux.T @ B @ Uy
    X_hat = np.zeros_like(B_hat)

    for i in range(k):
        for j in range(k):
            X_hat[i, j] = B_hat[i, j] / (Sx[i] * Sy[j])

    return VxT.T @ X_hat @ VyT

def tikhonov_2d(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha):
    B_hat = Ux.T @ B @ Uy
    X_hat = np.zeros_like(B_hat)

    for i in range(len(Sx)):
        for j in range(len(Sy)):
            lam2 = (Sx[i] * Sy[j]) ** 2
            X_hat[i,j] = (lam2 / (lam2 + alpha)) * B_hat[i, j] / (Sx[i] * Sy[j])

    return VxT.T @ X_hat @ VyT
 """

def tsvd_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, k):
    B_hat = Ux.T @ B @ Uy
    X_hat = np.zeros_like(B_hat)

    denom = np.outer(Sx[:k], Sy[:k])
    X_hat[:k, :k] = B_hat[:k, :k] / denom

    return VxT.T @ X_hat @ VyT

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


k = 30
alpha = 1e-3

x_tsvd = tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, blurred, k)
x_tikhonov = tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, blurred, alpha)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(x_tsvd, cmap='gray')
plt.title("TSVD reconstruction")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(x_tikhonov, cmap='gray')
plt.title("Tikhonov reconstruction")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img, cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.show()

# Quantitative comparison for relative errors between TSVD and Tikhonov
def relatative_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

print("TSVD relative error: ", relatative_error(x_tsvd, img))
print("Tikhonov relative error: ", relatative_error(x_tikhonov, img))

# Computational timing comparison between TSVD and Tikhonov
start = time.time()
tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, blurred, k)
tsvd_time = time.time() - start

start = time.time()
tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, blurred, alpha)
tikhonov_time = time.time() - start

print("TSVd time: ", tsvd_time)
print("Tikhonov time: ", tikhonov_time)



sizes = [32, 48, 64, 96,128]
k_frac = 0.1
alpha = 1e-2

tsvd_times = []
tikhonov_times = []
psnr_tsvd, psnr_tikh = [], []
ssim_tsvd, ssim_tikh = [], []

for n in sizes:
    m = n
    img = img_as_float(data.camera())
    img = resize(img, (n, m), anti_aliasing=True)

    Ax = gaussian_blur_1d(n, sigma)
    Ay = gaussian_blur_1d(m, sigma)

    B = Ax @ img @ Ay.T
    B += 0.01 * np.random.randn(*B.shape)

    Ux, Sx, VxT = np.linalg.svd(Ax, full_matrices=False)
    Uy, Sy, VyT = np.linalg.svd(Ay, full_matrices=False)

    k = int(k_frac * n)

    start = time.time()
    x_tsvd = tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, B, k)
    tsvd_times.append(time.time() - start)
    psnr_tsvd.append(psnr(img, x_tsvd))
    ssim_tsvd.append(ssim(img, x_tsvd, data_range=1.0))

    start = time.time()
    x_tikhonov = tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha)
    tikhonov_times.append(time.time() - start)
    psnr_tikh.append(psnr(img, x_tikhonov))
    ssim_tikh.append(ssim(img, x_tikhonov, data_range=1.0))

plt.figure()
plt.plot(sizes, tsvd_times, 'o-', label='TSVD (low-rank)')
plt.plot(sizes, tikhonov_times, 's-', label='Tikhonov (optimised)')
plt.xlabel("Image size (nxn)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime scaling (2D separable blur)")
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame({
    "Size": sizes,
    "PSNR TSVD": psnr_tsvd,
    "PSNR_Tikhonov":psnr_tikh,
    "SSIM TSVD": ssim_tsvd,
    "SSIM Tikhonov":ssim_tikh
})

print(df)

# artifacts comparison plots
n = 128
m = 128
img = img_as_float(data.camera())
img = resize(img, (n, m), anti_aliasing=True)

Ax = gaussian_blur_1d(n, sigma)
Ay = gaussian_blur_1d(m, sigma)

B = Ax @ img @ Ay.T
B += 0.01 * np.random.randn(*B.shape)

Ux, Sx, VxT = np.linalg.svd(Ax, full_matrices=False)
Uy, Sy, VyT = np.linalg.svd(Ay, full_matrices=False)

k = int(0.1 * n)

x_tsvd = tsvd_2d_lowrank(Ux, Sx, VxT, Uy, Sy, VyT, B, k)
x_tikhonov = tikhonov_2d_optimised(Ux, Sx, VxT, Uy, Sy, VyT, B, alpha)

plt.figure(figsize=(12,4))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(B, cmap='gray')
plt.title("Blurred + Noise")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(x_tsvd, cmap='gray')
plt.title("TSVD")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(x_tikhonov, cmap='gray')
plt.title("Tikhonov")
plt.axis("off")

plt.tight_layout()
plt.show()
