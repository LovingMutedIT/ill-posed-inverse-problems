import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

goce = np.load("../datasets/goce_egg_nom_2i_day.npz")

positions = goce["position"]
gradients = goce["gradients"]

D_obs = gradients[:, 2]
print("Samples:", len(D_obs))

# converting satellite track to regular grid
def ecef_to_spherical(xyz):
    # using an overflow-safe norm
    scale = np.max(np.abs(xyz), axis=1) + 1e-20
    xs = xyz / scale[:, None]
    r = scale * np.sqrt(np.sum(xs * xs, axis=1))

    z_over_r = xyz[:, 2] / r
    z_over_r = np.clip(z_over_r, -1.0, 1.0)

    lat = np.arcsin(z_over_r)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return lat, lon

lat, lon = ecef_to_spherical(positions)

R = 6371e3
n = 128
x = R * lon * np.cos(lat.mean())
y = R * lat

xi = np.linspace(x.min(), x.max(), n)
yi = np.linspace(y.min(), y.max(), n)
LAT, LON = np.meshgrid(xi, yi)

D_grid = griddata(
    (lat, lon),
    D_obs,
    (LAT, LON),
    method="linear",
    fill_value=0.0
)

dx = xi[1] - xi[0]
# implementing forward operator first (fft-based)

# building the frequency grid
def frequency_grid(n, dx=1.0):
    kx = np.fft.fftfreq(n, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(n, d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    return K

def gravity_forward_operator(X, h, dx=1.0):
    # here X: gravity potential at Earth's surface (2D) and h: Satellite altitude
    n = X.shape[0]
    K = frequency_grid(n, dx=1.0)

    # Singular values (physics-based)
    sigma = (K**2) * np.exp(-K * h)
    sigma[0, 0] = 0.0

    X_hat = np.fft.fft2(X)
    D_hat = sigma * X_hat

    D = np.real(np.fft.ifft2(D_hat))
    return D, sigma

# adding coloured noise (goce-like)
def coloured_noise(shape, beta=1.5):
    noise = np.random.randn(*shape)
    F = np.fft.fft2(noise)
    fx = np.fft.fftfreq(shape[0])[:, None]
    fy = np.fft.fftfreq(shape[1])[None, :]
    spectrum = (fx**2 + fy**2 + 1e-6)**(-beta/2)
    return np.real(np.fft.ifft2(F * spectrum))


def tsvd_inversion(D, sigma, tau):
    D_hat = np.fft.fft2(D)
    mask = sigma > tau
    X_hat = np.zeros_like(D_hat)
    X_hat[mask] = D_hat[mask] / sigma[mask]
    return np.real(np.fft.ifft2(X_hat))

def tikhonov_inversion(D, sigma, alpha):
    D_hat = np.fft.fft2(D)
    X_hat = (sigma / (sigma**2 + alpha)) * D_hat

    return np.real(np.fft.ifft2(X_hat))

def l_curve(D, sigma, alphas):
    res_norms = []
    sol_norms = []

    D_hat = np.fft.fft2(D)
    
    for alpha in alphas:
        X_hat = (sigma / (sigma**2 + alpha)) * D_hat
        residual = (alpha / (sigma**2 + alpha)) * D_hat

        res_norms.append(np.sqrt(np.sum(np.abs(residual)**2) + 1e-20))
        sol_norms.append(np.sqrt(np.sum(np.abs(X_hat)**2) + 1e-20))

    return np.array(res_norms), np.array(sol_norms)

def lcurve_corner(res, sol):
    eps = 1e-12
    log_r = np.log(res + eps)
    log_s = np.log(sol + eps)

    d1r = np.gradient(log_r)
    d1s = np.gradient(log_s)
    d2r = np.gradient(d1r)
    d2s = np.gradient(d1s)


    denom = (d1r**2 + d1s**2)**1.5 + eps
    curvature = np.abs(d1r * d2s - d1s * d2r) / denom

    curvature[~np.isfinite(curvature)] = 0.0
    return curvature

def gcv_score(D, sigma, alphas):
    D_hat = np.fft.fft2(D)
    scores = []

    for a in alphas:
        num = np.sum((a / (sigma**2 + a))**2 * np.abs(D_hat)**2)
        trace = np.sum(a / (sigma**2 + a))
        scores.append(num / trace**2)


    return np.array(scores)

def gravity_singular_values(n, h, dx=1.0):
    kx = np.fft.fftfreq(n, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(n, d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    sigma = K**2 * np.exp(-K * h)
    sigma[0, 0] = 0.0
    return sigma

def synthetic_potential(n, dx):
    x = np.linspace(-n//2, n//2, n) * dx
    X, Y = np.meshgrid(x, x)

    X_true = (
        np.exp(-((X-20e3)**2 + (Y-10e3)**2) / (2*(30e3)**2)) + 0.7 * np.exp(-((X+30e3)**2 + (Y+20e3)**2) / (2*(20e3)**2))
    )

    return X_true

h = 250e3
X_true = synthetic_potential(n, dx)

D_clean, sigma = gravity_forward_operator(X_true, h, dx)
D_noisy = D_clean + 0.05 * np.std(D_clean) * coloured_noise(D_clean.shape, beta=1.5)

alphas = np.logspace(-6, 2, 50)

res, sol = l_curve(D_noisy, sigma, alphas)
curv = lcurve_corner(res, sol)
gcv_scores = gcv_score(D_noisy, sigma, alphas)

alpha_lc = alphas[np.argmax(curv)]
alpha_gcv = alphas[np.argmin(gcv_scores)]

X_lc = tikhonov_inversion(D_noisy, sigma, alpha_lc)
X_gcv = tikhonov_inversion(D_noisy, sigma, alpha_gcv)

def rmse(x_ref, x):
    # using kahan-style scaling
    diff = x - x_ref
    scale = np.max(np.abs(diff)) + 1e-20
    return scale * np.sqrt(np.mean(diff / scale)**2)

print("RMSE L-curve:", rmse(X_true, X_lc))
print("RMSE GCV: ", rmse(X_true, X_gcv))

print(alpha_lc, alpha_gcv)

""" sigma = gravity_singular_values(n, h, dx)

alphas = np.logspace(-6, 2, 50)

res, sol = l_curve(D_grid, sigma, alphas)
gcv_scores = gcv_score(D_grid, sigma, alphas)

curv = lcurve_corner(res, sol)

alpha_lc = alphas[np.argmax(curv)]
alpha_gcv = alphas[np.argmin(gcv_scores)]

x_lc = tikhonov_inversion(D_grid, sigma, alpha_lc)
x_gcv = tikhonov_inversion(D_grid, sigma, alpha_gcv)
 """

def radial_spectrum(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(X)))**2
    F[F > 1e10 * np.median(F)] = 0.0
    n = X.shape[0]
    y, x = np.indices((n, n))
    r = np.sqrt((x - n/2)**2 + (y - n/2)**2).astype(int)

    spec = np.bincount(r.ravel(), F.ravel())
    count = np.bincount(r.ravel())
    return spec / (count + 1e-20)


# comparison between true spectrum and synthetic spectrum
spec_true = radial_spectrum(X_true)
spec_rec = radial_spectrum(X_lc)

spec_true /= spec_true.max()
spec_rec /= spec_rec.max()

kmax = 40
plt.semilogy(spec_true[:kmax], label="True Potential")
plt.semilogy(spec_rec[:kmax], label="Reconstructed potential")
plt.xlabel("Radial wavenumber")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.show()

sigma_goce = gravity_singular_values(n, h, dx)

alphas_goce = np.logspace(-8, 0, 60)

res_goce, sol_goce = l_curve(D_grid, sigma_goce, alphas_goce)
curv_goce = lcurve_corner(res_goce, sol_goce)
gcv_goce = gcv_score(D_grid, sigma_goce, alphas_goce)

alpha_goce_lc = alphas_goce[np.argmax(curv_goce)]
alpha_goce_gcv = alphas_goce[np.argmin(gcv_goce)]

D_goce_spectrum = radial_spectrum(D_grid)

print("GOCE alpha (L-curve):", alpha_goce_lc)
print("GOCE alpha (GCV):", alpha_goce_gcv)


""" plt.figure(figsize=(10,4))

plt.subplot(1,2,2)
plt.imshow(X_goce_gcv, cmap="viridis")
plt.title("GOCE Gravity Potential (GCV)")
plt.colorbar()

plt.tight_layout()
plt.show()
 """

""" def radial_spectrum(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(X)))**2
    F[F > 1e10 * np.median(F)] = 0.0
    n = X.shape[0]
    y, x = np.indices((n, n))
    r = np.sqrt((x - n/2)**2 + (y - n/2)**2).astype(int)

    spec = np.bincount(r.ravel(), F.ravel())
    count = np.bincount(r.ravel())
    return spec / (count + 1e-20)
 """
def sanitize(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[np.abs(X) > 1e6] = 0.0
    return X

def normalize(spec):
    return spec / (np.max(spec) + 1e-20)

kmax = 40

spec_gt = radial_spectrum(X_true)
spec_syn = radial_spectrum(X_lc)

spec_gt = normalize(spec_gt)
spec_syn = normalize(spec_syn)

plt.loglog(spec_gt[:kmax], label="Synthetic GT")
plt.loglog(spec_syn[:kmax], label="Synthetic Recon")

plt.xlabel("Radial wavenumber")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.show()

print("Synthetic GT spectrum min/max:", spec_gt.min(), spec_gt.max())
print("Synthietic Recon spectrum min/max:", spec_syn.min(), spec_syn.max())
