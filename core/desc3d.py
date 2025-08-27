import numpy as np
import scipy
import math
from typing import List
import numpy.linalg as la
from scipy.special import legendre

def real_spherical_harmonics(m: int, l: int, phi: float, theta: float) -> float:
    """
    Compute real form of the Spherical harmonics.
    See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form for details.

    Args:
        m (int): Order of the harmonic.
        l (int): Degree of the harmonic.
        phi (float): Azimuthal angle.
        theta (float): Polar angle.

    Returns:
        float: Real-valued spherical harmonic value.
    """
    if m > 0:
        return scipy.special.sph_harm_y(l, m, phi, theta).real * ((-1) ** m) / math.sqrt(2)
    elif m == 0:
        return scipy.special.sph_harm_y(l, m, phi, theta).real
    else:
        return scipy.special.sph_harm_y(l, -m, phi, theta).imag * ((-1) ** m) / math.sqrt(2)


def pca_spherical_harmonics(points: np.ndarray, max_l: int = 3) -> np.ndarray:
    """
    Compute real spherical harmonics coefficients aligned with PCA orientation.
    Based on "Toward High Throughput 3D Virtual Screening Using Spherical Harmonic Surface".
    See https://pubs.acs.org/doi/abs/10.1021/ci7001507 and https://arxiv.org/pdf/1509.09211 for details.
    Parameters:
        points (np.ndarray): (N, 3) array of 3D coordinates.
        max_l (int): Maximum degree `l` of spherical harmonics (default 3).

    Returns:
        np.ndarray: Array of length max_l^2 with real SH coefficients.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")

    points = points - points.mean(axis=0)
    n_points = len(points)

    # PCA alignment
    cov = np.cov(points.T)
    eigvecs = la.eig(cov)[1].T  # eigenvectors as rows

    for i in range(2):  # flip axes for consistent sign orientation
        proj = points @ eigvecs[i]
        if -proj.min() > proj.max():
            eigvecs[i] = -eigvecs[i]
    if la.det(eigvecs) < 0:
        eigvecs[2] = -eigvecs[2]

    aligned = points @ eigvecs.T
    x, y, z = aligned.T
    r = la.norm(aligned, axis=1)
    r[r == 0] = 1e-6  # Avoid zero radius

    # Spherical coordinates
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    sin_theta = np.sin(theta)
    sin_theta[sin_theta == 0] = 1e-6
    phi = np.arccos(np.clip(x / (r * sin_theta), -1.0, 1.0))
    phi = np.where(y < 0, -phi, phi)

    # Compute real spherical harmonic coefficients
    num_coeffs = max_l ** 2
    descriptors = np.zeros(num_coeffs)
    coeff_idx = 0

    for l in range(max_l):
        for m in range(-l, l + 1):
            y_lm = np.array([real_spherical_harmonics(m, l, phi[i], theta[i]) for i in range(n_points)])
            denom = np.sum(y_lm * y_lm) or 1e-6
            descriptors[coeff_idx] = np.dot(r, y_lm) / denom
            coeff_idx += 1
    print(descriptors)
    return descriptors


def pca_spherical_harmonics_mse(points: np.ndarray, max_l: int = 3) -> np.ndarray:
    """
    Compute real spherical harmonics coefficients aligned with PCA orientation.
    Based on "Toward High Throughput 3D Virtual Screening Using Spherical Harmonic Surface".
    See https://pubs.acs.org/doi/abs/10.1021/ci7001507 and https://arxiv.org/pdf/1509.09211 for details.
    Parameters:
        points (np.ndarray): (N, 3) array of 3D coordinates.
        max_l (int): Maximum degree `l` of spherical harmonics (default 3).

    Returns:
        np.ndarray: Array of length max_l^2 with real SH coefficients.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")

    points = points - points.mean(axis=0)
    n_points = len(points)

    # PCA alignment
    cov = np.cov(points.T)
    eigvecs = la.eig(cov)[1].T  # eigenvectors as rows

    for i in range(2):  # flip axes for consistent sign orientation
        proj = points @ eigvecs[i]
        if -proj.min() > proj.max():
            eigvecs[i] = -eigvecs[i]
    if la.det(eigvecs) < 0:
        eigvecs[2] = -eigvecs[2]

    aligned = points @ eigvecs.T
    x, y, z = aligned.T
    r = la.norm(aligned, axis=1)
    r[r == 0] = 1e-6  # Avoid zero radius

    # Spherical coordinates
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    sin_theta = np.sin(theta)
    sin_theta[sin_theta == 0] = 1e-6
    phi = np.arccos(np.clip(x / (r * sin_theta), -1.0, 1.0))
    phi = np.where(y < 0, -phi, phi)

    # Compute real spherical harmonic coefficients
    num_coeffs = max_l ** 2
    descriptors = np.zeros(num_coeffs)
    Y = np.zeros((n_points, num_coeffs))
    coeff_idx = 0
    for l in range(max_l):
        for m in range(-l, l + 1):
            Y[:, coeff_idx] = [
                real_spherical_harmonics(m, l, phi[i], theta[i]) for i in range(n_points)
            ]
            coeff_idx += 1

    # Solve least squares: minimize || Y a - r ||
    coeffs, _, _, _ = np.linalg.lstsq(Y, r, rcond=None)

    return coeffs

def rif_sh_descriptor(points: np.ndarray, harmonics: int = 8) -> np.ndarray:
    """
    Compute rotationally invariant descriptors using spherical harmonics (RIF-SH).
    Based on "Toward High Throughput 3D Virtual Screening Using Spherical Harmonic Surface".
    See https://pubs.acs.org/doi/abs/10.1021/ci7001507 for details.
    Parameters:
        points (np.ndarray): An (N, 3) array of 3D coordinates.
        harmonics (int): The maximum degree `l` of spherical harmonics to compute.

    Returns:
        np.ndarray: A (harmonics, ) array of A_l descriptors.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")

    points = points - points.mean(axis=0)
    r = la.norm(points, axis=1)
    r[r == 0] = 1e-6

    # Spherical coordinates
    theta = np.arccos(np.clip(points[:, 2] / r, -1.0, 1.0))
    sin_theta = np.sin(theta)
    sin_theta[sin_theta == 0] = 1e-6

    phi = np.arccos(np.clip(points[:, 0] / (r * sin_theta), -1.0, 1.0))
    phi = np.where(points[:, 1] < 0, -phi, phi)

    desc = np.zeros(harmonics)

    for l in range(harmonics):
        acc = 0.0
        for m in range(-l, l + 1):
            y_lm = np.array([scipy.special.sph_harm_y(l, m, phi[i], theta[i]) for i in range(len(points))])
            numerator = np.abs(np.dot(r, y_lm))
            denominator = np.sum(np.abs(y_lm * y_lm)) or 1e-6
            acc += (numerator / denominator) ** 2
        desc[l] = math.sqrt(acc)

    return desc
from scipy.special import eval_legendre

def cylindrical_harmonics_descriptor(points: np.ndarray, da: int = 3, dc: int = 2) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")

    points = points - points.mean(axis=0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    r = np.sqrt(y**2 + z**2)
    phi = np.arctan2(z, y)

    basis = []
    labels = []

    for l in range(-dc, dc + 1):
        if l >= 0:
            h_phi = np.cos(l * phi)
        else:
            h_phi = np.sin(-l * phi)

        for k in range(da + 1):
            Pk = eval_legendre(k, x)   # <-- FIXED
            basis.append(Pk * h_phi)
            labels.append((k, l))

    basis = np.vstack(basis).T
    coeffs_flat, _, _, _ = np.linalg.lstsq(basis, r, rcond=None)

    coeffs = np.zeros((da + 1, 2 * dc + 1))
    for idx, (k, l) in enumerate(labels):
        coeffs[k, l + dc] = coeffs_flat[idx]

    return np.ndarray.flatten(coeffs)


def bend_deformed_cylinder_descriptors(points: np.ndarray) -> List[float]:
    """
    Compute 8 BDC (Bent Deformed Cylinder) descriptors for a set of 3D points.
    Approximate shape as cylinder which was bent and deformed to have evolving ellipse in cross-section.
    See  https://arxiv.org/pdf/1509.09211 for details.

    Parameters:
        points (np.ndarray): An (N, 3) array of 3D coordinates.

    Returns:
        List[float]: A list of 8 shape descriptors representing width, bending, and evolving elliptical cross-section.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")
    if len(points) < 3:
        return [0.0] * 8  # Not enough points to compute descriptors

    # Center the data
    centered = points - points.mean(axis=0)

    # Principal Component Analysis (PCA)
    cov = np.cov(centered.T)
    eigvals, eigvecs = la.eigh(cov)  # eigh is safer for symmetric covariance
    eigvecs = eigvecs[:, ::-1].T  # sort descending, rows as eigenvectors

    # Orient the axes consistently (flip directions based on extrema)
    for i in range(2):
        proj = centered @ eigvecs[i]
        if -proj.min() > proj.max():
            eigvecs[i] = -eigvecs[i]
    if la.det(eigvecs) < 0:
        eigvecs[2] = -eigvecs[2]

    # Project points into PCA-aligned frame
    rotated = centered @ eigvecs.T
    x, y, z = rotated.T

    # Width along x (principal axis)
    x_min, x_max = x.min(), x.max()
    width = max(x_max - x_min, 1e-10)  # avoid 0 width

    # Normalize x to [-1, 1]
    a = 2 / width
    b = 1 - a * x_max
    x_scaled = a * x + b

    # Orthogonal quadratic polynomial: 3x^2 - 1
    xp = 3 * x_scaled**2 - 1
    denom = np.sum(xp**2)
    if denom < 1e-12:
        xp = np.zeros_like(x_scaled)
    else:
        xp /= denom

    # Coefficients for bending parabola
    cy = y @ xp
    cz = z @ xp
    bend = math.hypot(cy, cz)
    if bend < 1e-12:
        bend = 1e-10
        cy, cz = 1.0, 0.0  # arbitrary orientation if no bend
    else:
        cy /= bend
        cz /= bend

    # Rotate yz-plane to align bending with y-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0,  cy, -cz],
        [0,  cz,  cy]
    ])
    aligned = rotated @ rotation_matrix.T

    # Remove the bending component
    x_final = x_scaled
    y_final = aligned[:, 1] - bend * (3 * x_scaled**2 - 1)
    z_final = aligned[:, 2]

    # Cross-section features
    y2 = y_final**2
    z2 = z_final**2
    yz = y_final * z_final

    y2_mean = y2.mean() if y2.size else 0.0
    z2_mean = z2.mean() if z2.size else 0.0
    yz_mean = yz.mean() if yz.size else 0.0

    # Linear evolution (correlation with x)
    denom_lin = np.sum(x_final**2)
    if denom_lin < 1e-12:
        lin_weight = np.zeros_like(x_final)
    else:
        lin_weight = x_final / denom_lin

    y2_lin = lin_weight @ y2
    z2_lin = lin_weight @ z2
    yz_lin = lin_weight @ yz

    return [float(width), float(2 * bend), float(y2_mean), float(z2_mean),
            float(yz_mean), float(y2_lin), float(z2_lin), float(yz_lin)]


def ultrafast_shape_recognition_descriptors(points: np.ndarray) -> List[float]:
    """
    Compute 12 USR (Ultrafast Shape Recognition) descriptors for a given set of 3D points.
    Based on "Ultrafast shape recognition for similarity search in molecular database".
    See https://rspa.royalsocietypublishing.org/content/463/2081/1307 for details.

    Parameters:
        points (np.ndarray): An (N, 3) array of 3D coordinates.

    Returns:
        List[float]: A list of 12 USR shape descriptors.
    """
    def moments(distances: np.ndarray) -> List[float]:
        """Compute mean, variance, and skewness (kurtosis) of a 1D array."""
        mean = distances.mean()
        var = distances.var()
        skew = np.mean(((distances - mean) / math.sqrt(var))**3) if var > 0 else 0.0
        return [mean, var, skew]

    def get_extreme_point(points_array: np.ndarray, reference: np.ndarray, mode: str = 'closest') -> np.ndarray:
        """Return the closest or farthest point to the reference."""
        dists = la.norm(points_array - reference, axis=1)
        idx = np.argmin(dists) if mode == 'closest' else np.argmax(dists)
        return points_array[idx]

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a (N, 3) array of 3D points.")

    # First point: centroid
    centroid = points.mean(axis=0)
    desc = moments(la.norm(points - centroid, axis=1))

    # Second point: closest atom to centroid
    closest = get_extreme_point(points, centroid, 'closest')
    desc += moments(la.norm(points - closest, axis=1))

    # Third point: farthest atom from closest
    farthest_from_closest = get_extreme_point(points, closest, 'farthest')
    desc += moments(la.norm(points - farthest_from_closest, axis=1))

    # Fourth point: farthest atom from previous
    farthest_from_third = get_extreme_point(points, farthest_from_closest, 'farthest')
    desc += moments(la.norm(points - farthest_from_third, axis=1))
    print(desc)
    return desc
