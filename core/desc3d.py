import numpy as np
from numpy import linalg as la
import math
import scipy
from typing import List

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

    return descriptors

def rif_sh_descriptors(points: np.ndarray, harmonics: int = 8) -> np.ndarray:
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

    # Center the data
    centered = points - points.mean(axis=0)

    # Principal Component Analysis (PCA)
    cov = np.cov(centered.T)
    eigvecs = la.eig(cov)[1].T  # Transpose to get eigenvectors as rows

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
    width = x_max - x_min

    # Normalize x to [-1, 1]
    a = 2 / width
    b = 1 - a * x_max
    x_scaled = a * x + b

    # Orthogonal quadratic polynomial: 3x^2 - 1
    xp = 3 * x_scaled**2 - 1
    xp /= np.sum(xp**2)

    # Coefficients for bending parabola
    cy = y @ xp
    cz = z @ xp
    bend = math.hypot(cy, cz) or 1e-10  # Prevent division by zero
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

    y2_mean = y2.mean()
    z2_mean = z2.mean()
    yz_mean = yz.mean()

    # Linear evolution (correlation with x)
    lin_weight = x_final / np.sum(x_final**2)
    y2_lin = lin_weight @ y2
    z2_lin = lin_weight @ z2
    yz_lin = lin_weight @ yz

    return [width, 2 * bend, y2_mean, z2_mean, yz_mean, y2_lin, z2_lin, yz_lin]

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

    return desc
