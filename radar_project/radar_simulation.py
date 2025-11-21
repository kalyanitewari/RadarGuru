"""
radar_simulation.py

Helper functions to generate synthetic 1D/2D radar-like signals and heatmaps.
Provides dataset generation utilities used by the notebooks.

Functions:
- generate_range_doppler_heatmap(...)
- generate_range_profile(...)
- generate_doppler_spectrum(...)
- apply_fft(...)
- visualize_heatmap(...)
- generate_dataset(...)
- denoise_background_subtract(...)

"""

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Small RNG helper
_rng = np.random.default_rng(42)


def _gaussian_blob(shape, center, sigma, amplitude=1.0):
    """Return a 2D gaussian blob array of given shape."""
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    xx, yy = np.meshgrid(x, y)
    cx, cy = center
    g = amplitude * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma ** 2))
    return g


def generate_range_doppler_heatmap(range_bins=128, doppler_bins=64,
                                    scenario='empty', metal=False,
                                    clutter_level=0.1, snr_db=20):
    """
    Create a synthetic range-Doppler 2D heatmap.

    Params:
    - range_bins, doppler_bins: output shape
    - scenario: 'empty', 'metal_object', 'clutter', 'hidden'
    - metal: whether the main target is metallic (stronger reflection)
    - clutter_level: 0..1 amplitude multiplier for random clutter
    - snr_db: approximate SNR for target vs noise

    Returns:
    - heatmap: 2D float array (range_bins x doppler_bins)
    """
    shape = (range_bins, doppler_bins)
    heatmap = np.zeros(shape, dtype=float)

    # background clutter: low-level speckle
    if scenario in ('clutter', 'hidden') or clutter_level > 0:
        clutter = _rng.normal(scale=clutter_level, size=shape)
        heatmap += np.abs(clutter)

    # Add 0..2 objects
    n_objects = 0
    if scenario == 'empty':
        n_objects = 0
    elif scenario == 'metal_object':
        n_objects = 1
    elif scenario == 'clutter':
        n_objects = _rng.integers(1, 3)
    elif scenario == 'hidden':
        # object is present but partially occluded by clutter; still place it
        n_objects = 1

    for i in range(n_objects):
        # pick a random center within array
        cx = _rng.uniform(0.1 * shape[1], 0.9 * shape[1])
        cy = _rng.uniform(0.1 * shape[0], 0.9 * shape[0])
        sigma = _rng.uniform(1.5, min(shape) * 0.06)
        amplitude = _rng.uniform(0.6, 1.0)
        if metal:
            amplitude *= _rng.uniform(1.5, 2.5)
        blob = _gaussian_blob(shape, center=(cx, cy), sigma=sigma, amplitude=amplitude)
        heatmap += blob

    # Optionally add a faint metallic signature spread across doppler for 'hidden'
    if scenario == 'hidden' and metal:
        # add a weak elongated feature
        rpos = int(shape[0] * 0.5)
        heatmap[rpos:rpos + 2, :] += 0.5 * np.exp(-np.linspace(-2, 2, shape[1]) ** 2)[None, :]

    # Add thermal noise
    noise_std = 10 ** (-snr_db / 20.0)
    heatmap += np.abs(_rng.normal(scale=noise_std, size=shape))

    # Normalize
    heatmap = heatmap / (np.max(heatmap) + 1e-12)

    # Apply mild blur to resemble sensor PSF
    heatmap = ndimage.gaussian_filter(heatmap, sigma=1.0)
    heatmap = heatmap / (np.max(heatmap) + 1e-12)
    return heatmap


def apply_fft(signal_2d):
    """Apply 2D FFT magnitude (useful to simulate transforms). Returns log-magnitude."""
    f = np.fft.fftshift(np.fft.fft2(signal_2d))
    mag = np.abs(f)
    mag = np.log1p(mag)
    mag = mag / (mag.max() + 1e-12)
    return mag


def visualize_heatmap(heatmap, title=None, cmap='inferno', show=True, savepath=None):
    plt.figure(figsize=(5, 4))
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='Normalized amplitude')
    if title:
        plt.title(title)
    plt.xlabel('Doppler bins')
    plt.ylabel('Range bins')
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def generate_dataset(out_dir, n_samples=300, img_size=(64, 64), seed=42):
    """
    Generate a dataset of synthetic heatmaps and labels.
    Saves .npy arrays and PNG preview images, plus a labels.csv.

    Produces balanced metal / non-metal samples.

    Returns: list of (filepath, label)
    """
    _rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, 'images')
    npy_dir = os.path.join(out_dir, 'npy')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    half = n_samples // 2
    records = []
    for i in range(n_samples):
        if i < half:
            label = 1  # metal
            scenario = _rng.choice(['metal_object', 'clutter'], p=[0.7, 0.3])
            clutter = _rng.uniform(0.0, 0.2)
        else:
            label = 0  # non-metal
            scenario = _rng.choice(['empty', 'clutter'], p=[0.4, 0.6])
            clutter = _rng.uniform(0.05, 0.4)
        heatmap = generate_range_doppler_heatmap(range_bins=img_size[0],
                                                 doppler_bins=img_size[1],
                                                 scenario=scenario,
                                                 metal=(label == 1),
                                                 clutter_level=clutter,
                                                 snr_db=int(_rng.uniform(6, 25)))
        # Save .npy
        npy_path = os.path.join(npy_dir, f'sample_{i:04d}.npy')
        np.save(npy_path, heatmap.astype(np.float32))
        # Save PNG preview
        png_path = os.path.join(images_dir, f'sample_{i:04d}.png')
        visualize_heatmap(heatmap, title=f'label={label} idx={i}', show=False, savepath=png_path)
        records.append((npy_path, png_path, label))
    # Save labels CSV
    import csv
    csv_path = os.path.join(out_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['npy_path', 'png_path', 'label'])
        for r in records:
            writer.writerow(r)
    return {'records': records, 'csv': csv_path}


def denoise_background_subtract(heatmap, method='median', kernel_size=5):
    """
    Simple background subtraction and denoising.
    - method='median' subtract median filtered background
    - method='gaussian' subtract gaussian-smoothed background
    Returns processed heatmap clipped 0..1
    """
    if method == 'median':
        bg = ndimage.median_filter(heatmap, size=(kernel_size, kernel_size))
    else:
        bg = ndimage.gaussian_filter(heatmap, sigma=kernel_size / 3.0)
    proc = heatmap - bg
    proc = proc - proc.min()
    if proc.max() > 0:
        proc = proc / proc.max()
    return proc

def generate_range_profile(range_bins=128, scenario='empty', metal=False, clutter_level=0.1, snr_db=20):
    """Generate a 1D synthetic range profile.

    Creates Gaussian echoes at certain ranges plus noise/clutter.

    Returns: 1D array length range_bins normalized 0..1
    """
    profile = np.zeros(range_bins, dtype=float)
    # clutter baseline
    if clutter_level > 0 or scenario in ('clutter', 'hidden'):
        profile += np.abs(_rng.normal(scale=clutter_level * 0.3, size=range_bins))
    n_targets = 0
    if scenario == 'metal_object':
        n_targets = 1
    elif scenario == 'clutter':
        n_targets = _rng.integers(1, 3)
    elif scenario == 'hidden':
        n_targets = 1
    for _ in range(n_targets):
        center = _rng.uniform(0.1 * range_bins, 0.9 * range_bins)
        sigma = _rng.uniform(range_bins * 0.01, range_bins * 0.05)
        amp = _rng.uniform(0.6, 1.0)
        if metal:
            amp *= _rng.uniform(1.5, 2.5)
        x = np.arange(range_bins)
        profile += amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    noise_std = 10 ** (-snr_db / 20.0)
    profile += np.abs(_rng.normal(scale=noise_std, size=range_bins))
    if profile.max() > 0:
        profile /= profile.max()
    return profile

def generate_doppler_spectrum(doppler_bins=64, scenario='empty', metal=False, clutter_level=0.1, snr_db=20):
    """Generate a 1D synthetic Doppler spectrum.

    Simulates velocity distribution with potential metallic target peaks.
    Returns normalized 1D array length doppler_bins.
    """
    spec = np.zeros(doppler_bins, dtype=float)
    if clutter_level > 0 or scenario in ('clutter', 'hidden'):
        spec += np.abs(_rng.normal(scale=clutter_level * 0.25, size=doppler_bins))
    n_targets = 0
    if scenario == 'metal_object':
        n_targets = 1
    elif scenario == 'clutter':
        n_targets = _rng.integers(1, 3)
    elif scenario == 'hidden':
        n_targets = 1
    for _ in range(n_targets):
        center = _rng.uniform(0.15 * doppler_bins, 0.85 * doppler_bins)
        sigma = _rng.uniform(doppler_bins * 0.01, doppler_bins * 0.07)
        amp = _rng.uniform(0.5, 1.0)
        if metal:
            amp *= _rng.uniform(1.7, 2.8)
        x = np.arange(doppler_bins)
        spec += amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    noise_std = 10 ** (-snr_db / 20.0)
    spec += np.abs(_rng.normal(scale=noise_std, size=doppler_bins))
    if spec.max() > 0:
        spec /= spec.max()
    return spec

def extract_features(raw_heatmap, processed_heatmap=None, k_top=10):
    """Extract a lightweight feature vector from a raw (and optional processed) heatmap.

    Features (concatenated):
    - raw: mean, std, max, energy (sum of squares), fraction pixels > 0.5
    - processed (if provided): same stats
    - top-k intensities from processed (or raw if processed missing)
    Returns 1D numpy array.
    """
    raw = raw_heatmap.astype(float)
    proc = processed_heatmap.astype(float) if processed_heatmap is not None else None
    def stats(a):
        return [a.mean(), a.std(), a.max(), np.sum(a*a), np.mean(a > 0.5)]
    feats = stats(raw)
    if proc is not None:
        feats += stats(proc)
        base = proc
    else:
        # duplicate placeholder for processed stats to keep dimensionality consistent
        feats += [0.0]*5
        base = raw
    # top-k intensities
    flat = np.sort(base.ravel())[::-1]
    k = min(k_top, flat.shape[0])
    feats += flat[:k].tolist()
    # pad if fewer than k_top
    if k < k_top:
        feats += [0.0]*(k_top - k)
    return np.array(feats, dtype=float)

if __name__ == '__main__':
    # Non-blocking demo CLI
    import argparse, sys
    parser = argparse.ArgumentParser(description='Synthetic radar heatmap demo')
    parser.add_argument('--range-bins', type=int, default=128)
    parser.add_argument('--doppler-bins', type=int, default=64)
    parser.add_argument('--scenario', type=str, default='metal_object', choices=['empty','metal_object','clutter','hidden'])
    parser.add_argument('--metal', action='store_true', help='Treat main target as metal for stronger return')
    parser.add_argument('--clutter', type=float, default=0.05, help='Clutter level multiplier')
    parser.add_argument('--snr-db', type=int, default=20)
    parser.add_argument('--show', action='store_true', help='Show interactive windows (may block).')
    parser.add_argument('--out-dir', type=str, default='data/demo', help='Directory to save demo images.')
    args = parser.parse_args()

    # Auto-disable show if running in a non-interactive environment unless --show explicitly passed
    auto_show = args.show and sys.stdout.isatty()

    os.makedirs(args.out_dir, exist_ok=True)
    hm = generate_range_doppler_heatmap(args.range_bins, args.doppler_bins,
                                        scenario=args.scenario, metal=args.metal,
                                        clutter_level=args.clutter, snr_db=args.snr_db)
    raw_path = os.path.join(args.out_dir, f'{args.scenario}_raw.png')
    visualize_heatmap(hm, title=f'Demo {args.scenario}', show=auto_show, savepath=raw_path)

    fft_img = apply_fft(hm)
    fft_path = os.path.join(args.out_dir, f'{args.scenario}_fft.png')
    visualize_heatmap(fft_img, title='FFT (log-magnitude)', show=auto_show, savepath=fft_path)

    print('Saved demo images:\n -', raw_path, '\n -', fft_path)
    if not auto_show and not args.show:
        print('Note: Use --show to open interactive windows (may block).')
