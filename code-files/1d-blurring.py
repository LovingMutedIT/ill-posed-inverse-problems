import numpy as np
import time
import matplotlib.pyplot as plt

# creating a 1-d blur operator for input signal
def blur_matrix(n, sigma=2):
    x = np.arange(n)
    A = np.zeros((n,n))
    for i in range(n):
        A[i] = np.exp(-(x-i)**2 / (2*sigma**2))
        # gaussian function is used to model blur
    A /= A.sum(axis=1, keepdims=True)
    return A


# generating synthetic data for input
n = 200
A = blur_matrix(n);

x_true = np.zeros(n)
x_true[50:150] = 1 # block signal

noise_level = 0.02
b = A @ x_true + noise_level * np.random.randn(n)

# computing SVD first (shared cost)
start = time.time()
U, S, VT = np.linalg.svd(A, full_matrices=False)
svd_time = time.time() - start


# Truncated SVD implementation definition
def tsvd_solution(U, S, VT, b, k):
    Uk = U[:,:k]
    Sk = S[:k]
    Vk = VT[:k, :]
    return Vk.T @ ((Uk.T @ b) / Sk)

# timing TSVD
ks = [5, 10, 20, 40, 80]
tsvd_times = []

for k in ks:
    start = time.time()
    xk = tsvd_solution(U, S, VT, b, k)
    tsvd_times.append(time.time() - start)


# Tikhonov implementation (SVD-based)
def tikhonov_solution(U, S, VT, b, alpha):
    filter = S**2 / (S**2 + alpha)
    return VT.T @ (filter * (U.T @ b) / S)

# timing Tikhonov
alphas = [1e-1, 1e-2, 1e-3, 1e-4]
tikhonov_times = []

for a in alphas:
    start = time.time()
    xa = tikhonov_solution(U, S, VT, b, a)
    tikhonov_times.append(time.time() - start)


# plotting results of deblurring comparison between tsvd and tikhonov
plt.figure(figsize=(10,5))
plt.plot(x_true, label="True", linewidth=2)

plt.plot(tsvd_solution(U,S,VT, b, 20), '--', label = "TSVD (k=20)")
plt.plot(tikhonov_solution(U, S, VT, b, 1e-3), ':', label = "Tikhonov (alpha = 1e-3)")

plt.legend()
plt.title("Deblurring TSVD vs Tikhonov")
plt.show()


print("SVD time: ", svd_time)
print("TSVD time: ", tsvd_times)
print("Tikhonov time: ", tikhonov_times)


## Algorithmic Time Complexity Analysis and Demonstration for TSVD vs Tikhonov Regalurization
sizes = [100, 200, 400, 800, 1200]
k_fraction = 0.1
alpha = 1e-3

tsvd_times = []
tikhonov_times = []

for n in sizes:
    print(f"Running n = {n}")

    A = blur_matrix(n)
    x_true = np.zeros(n)
    x_true[n//4:3*n//4] = 1
    b = A @ x_true + 0.01 * np.random.randn(n)

    # implementing full SVD only once
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    k = int(k_fraction*n)

    # TSVD timing calculations
    t0 = time.time()
    for _ in range(20):
        tsvd_solution(U, S, VT, b, k)
    tsvd_times.append((time.time() - t0) / 20)

    # Tikhonov Timing calculations
    t0 = time.time()
    for _ in range(20):
        tikhonov_solution(U, S, VT, b, alpha)
    tikhonov_times.append((time.time() - t0) / 20)


# plotting results for timing calculations (log-log complexity plots)
plt.figure(figsize = (7,5))
plt.loglog(sizes, tsvd_times, 'o-', label = "TSVD (O(nk))")
plt.loglog(sizes, tikhonov_times, 's-', label = "Tikhonov (O(n^2))")

plt.xlabel("Problem size n")
plt.ylabel("Average runtime (seconds)")
plt.title("Algorithmic Time Complexity")
plt.legend()
plt.grid(True, which="both")
plt.show()

# additionally verifying slope numerically for a stronger evidence
tsvd_slope = np.polyfit(np.log(sizes), np.log(tsvd_times), 1)[0]
tikhonov_slope = np.polyfit(np.log(sizes), np.log(tikhonov_times), 1)[0]

print("Estimated TSVD Slope: ", tsvd_slope)
print("Estimated Tikhonov slope: ", tikhonov_slope)

