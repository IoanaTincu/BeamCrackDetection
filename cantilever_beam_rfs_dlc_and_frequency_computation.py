import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Core reusable utilities
# ============================================================

def generate_positions(x_start, x_end, x_step):
    """
    Generate numerically clean beam positions using integer indexing,
    then rounding to the number of decimals implied by x_step.

    Example:
      x_step=0.002  -> decimals=3
      x_step=0.0001 -> decimals=4
    """
    span = x_end - x_start
    n_steps = int(round(span / x_step))

    # Ensure the endpoint aligns exactly with the chosen step size
    if not np.isclose(x_start + n_steps * x_step, x_end, atol=1e-12):
        raise ValueError("(x_end - x_start) / x_step must be an integer.")

    # Integer ticks avoid cumulative floating-point drift
    ticks = np.arange(n_steps + 1, dtype=int)
    x = x_start + ticks * x_step

    # Infer decimal places from step size (0.0001 -> 4)
    step_str = f"{x_step:.12f}".rstrip("0").rstrip(".")
    decimals = len(step_str.split(".")[1]) if "." in step_str else 0

    # Round to remove floating artifacts
    x = np.round(x, decimals=decimals)

    return x, decimals


def compute_normalized_curvature_squared(x, lambdas):
    """
    Compute squared normalized curvature along the beam for all modes.

    Returns
    -------
    squared_normalized_curvature : ndarray of shape (n_points, n_modes)
        Each column corresponds to one vibration mode.
        Values are in [0, 1] because they come from squared normalized curvature.
    """
    lambdas = np.asarray(lambdas, dtype=float)

    # Cantilever beam mode-shape coefficient:
    # ratio = (cos(λ) + cosh(λ)) / (sin(λ) + sinh(λ))
    ratio = (np.cos(lambdas) + np.cosh(lambdas)) / (
            np.sin(lambdas) + np.sinh(lambdas)
    )

    # Compute λx for all x locations and all modes
    lam_x = np.outer(x, lambdas)

    # Curvature expression (vectorized over x and modes)
    curvature = (
            -ratio * np.sin(lam_x)
            - ratio * np.sinh(lam_x)
            + np.cos(lam_x)
            + np.cosh(lam_x)
    )

    # Normalize curvature mode-wise by maximum absolute value
    normalized_curvature = -curvature / np.max(np.abs(curvature), axis=0)

    # Square the normalized curvature
    squared_normalized_curvature = normalized_curvature ** 2

    return squared_normalized_curvature


# ============================================================
# Healthy-beam frequency computation (single combined parameter)
# ============================================================

def compute_healthy_frequencies(lambdas, C):
    """
    Compute the natural frequencies of a HEALTHY beam using a single combined parameter C.

    Relationship:
        f_healthy,n = C * lambda_n^2

    where the combined parameter is defined as:
        C = sqrt(EI/(rho*A)) / (2*pi*L^2)

    Parameters
    ----------
    lambdas : ndarray
        Array of eigenvalues for cantilever beam modes (e.g., 10 values).
    C : float
        Combined constant capturing beam/material properties.

    Returns
    -------
    f_healthy : ndarray (n_modes,)
        Healthy natural frequencies for each mode (Hz).
    """
    lambdas = np.asarray(lambdas, dtype=float)
    return float(C) * (lambdas ** 2)


# ============================================================
# RFS + damaged frequency + delta frequency datasets
# ============================================================

def compute_rfs_dataframe(x, lambdas, severity):
    """
    Compute Relative Frequency Shift (RFS) along the beam for all modes at a given severity.

    Returns a DataFrame with columns:
        x, RFS_mode1..RFS_modeN, severity
    """
    snc2 = compute_normalized_curvature_squared(x, lambdas)

    # RFS = squared normalized curvature × severity
    rfs = snc2 * float(severity)

    columns = ["x"] + [f"RFS_mode{i}" for i in range(1, len(lambdas) + 1)] + ["severity"]
    data = np.column_stack([x, rfs, np.full_like(x, float(severity), dtype=float)])

    return pd.DataFrame(data, columns=columns)


def compute_damaged_frequency_dataframe(x, rfs, f_healthy):
    """
    Compute damaged frequencies per x and mode:
        f_damaged = f_healthy * (1 - RFS)

    Returns a DataFrame with columns:
        x, fdam_mode1..fdam_modeN
    """
    f_damaged = (1.0 - rfs) * f_healthy.reshape(1, -1)

    columns = ["x"] + [f"fdam_mode{i}" for i in range(1, f_healthy.size + 1)]
    data = np.column_stack([x, f_damaged])

    return pd.DataFrame(data, columns=columns)


def compute_delta_frequency_dataframe(x, rfs, f_healthy):
    """
    Compute delta frequency per x and mode:
        delta_f = f_healthy - f_damaged = f_healthy * RFS

    Returns a DataFrame with columns:
        x, deltaf_mode1..deltaf_modeN
    """
    delta_f = rfs * f_healthy.reshape(1, -1)

    columns = ["x"] + [f"deltaf_mode{i}" for i in range(1, f_healthy.size + 1)]
    data = np.column_stack([x, delta_f])

    return pd.DataFrame(data, columns=columns)


def generate_rfs_and_frequency_csvs(
        lambdas,
        severities,
        C,
        x_start=0.0,
        x_end=1.0,
        x_step=0.002,
        file_prefix="cantilever_beam_rfs_modes1to10",
        output_dir="single_crack_csvs",
        fdam_output_dir="single_crack_fdam_csvs",
        deltaf_output_dir="single_crack_deltaf_csvs",
        fdam_prefix="cantilever_beam_fdam_modes1to10",
        deltaf_prefix="cantilever_beam_deltaf_modes1to10",
):
    """
    Generate one CSV file per severity containing:
        x, RFS_mode1..RFS_modeN, severity

    EXTENDED:
    Also generate, for each severity:
        - damaged frequency CSV: x, fdam_mode1..fdam_modeN
        - delta frequency CSV:   x, deltaf_mode1..deltaf_modeN

    Notes
    -----
    - Uses compute_rfs_dataframe(...) to compute RFS.
    - Uses compute_healthy_frequencies(lambdas, C) once.
    - Uses compute_damaged_frequency_dataframe(...) and compute_delta_frequency_dataframe(...)
      for frequency outputs.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fdam_output_dir = Path(fdam_output_dir)
    fdam_output_dir.mkdir(parents=True, exist_ok=True)

    deltaf_output_dir = Path(deltaf_output_dir)
    deltaf_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Generate clean x values
    # --------------------------------------------------
    x, x_decimals = generate_positions(x_start, x_end, x_step)

    # --------------------------------------------------
    # Compute healthy frequencies once (vector length = n_modes)
    # f_healthy,n = C * lambda_n^2
    # --------------------------------------------------
    f_healthy = compute_healthy_frequencies(lambdas, C=C)

    written_files = []

    for sev in severities:
        # --------------------------------------------------
        # 1) Compute RFS DataFrame using your existing function
        # --------------------------------------------------
        df_rfs = compute_rfs_dataframe(x, lambdas, sev)

        # Extract RFS columns as a NumPy array for frequency computations
        rfs_cols = [c for c in df_rfs.columns if c.startswith("RFS_mode")]
        rfs = df_rfs[rfs_cols].to_numpy(dtype=float)  # shape: (n_points, n_modes)

        # --------------------------------------------------
        # 2) Compute frequency DataFrames using your existing functions
        # --------------------------------------------------
        df_fdam = compute_damaged_frequency_dataframe(x, rfs, f_healthy)
        df_deltaf = compute_delta_frequency_dataframe(x, rfs, f_healthy)

        # --------------------------------------------------
        # Filenames include severity tag like before
        # --------------------------------------------------
        sev_tag = f"{sev:.6g}".replace(".", "p").replace("-", "m")

        rfs_path = output_dir / f"{file_prefix}_sev_{sev_tag}.csv"
        fdam_path = fdam_output_dir / f"{fdam_prefix}_sev_{sev_tag}.csv"
        deltaf_path = deltaf_output_dir / f"{deltaf_prefix}_sev_{sev_tag}.csv"

        # --------------------------------------------------
        # Keep x visually clean; keep other floats in scientific notation
        # --------------------------------------------------
        for df_out in (df_rfs, df_fdam, df_deltaf):
            df_out["x"] = df_out["x"].map(lambda v: f"{float(v):.{x_decimals}f}")
            df_out.to_csv(
                (rfs_path if df_out is df_rfs else (fdam_path if df_out is df_fdam else deltaf_path)),
                index=False,
                float_format="%.12e",
            )

        written_files.extend([str(rfs_path), str(fdam_path), str(deltaf_path)])

        print(f"Saved CSV to: {rfs_path}")
        print(f"Saved CSV to: {fdam_path}")
        print(f"Saved CSV to: {deltaf_path}")

    return written_files


# ============================================================
# New DLC dataset generation
# ============================================================

def compute_dlc_dataframe(x, lambdas, eps=1e-15):
    """
    Compute Damage Location Coefficients (DLC) at each position x.

    Steps per row (i.e., per x):
    - compute squared normalized curvatures across modes: snc2(x, mode_i)
    - find max curvature over modes: max_i snc2(x, mode_i)
    - compute ratio for each mode: DLC_i(x) = snc2(x, mode_i) / max_i
      (eps prevents division by zero if max_i is extremely small)

    Returns a DataFrame with columns:
        x, DLC_mode1..DLC_modeN
    """
    snc2 = compute_normalized_curvature_squared(x, lambdas)

    # Row-wise maximum across modes (shape: n_points, 1)
    row_max = np.max(snc2, axis=1, keepdims=True)

    # Avoid division by zero (very unlikely, but safe)
    row_max = np.maximum(row_max, eps)

    # DLC ratios in [0, 1]
    dlc = snc2 / row_max

    columns = ["x"] + [f"DLC_mode{i}" for i in range(1, len(lambdas) + 1)]
    data = np.column_stack([x, dlc])

    return pd.DataFrame(data, columns=columns)


def generate_dlc_csv(
        lambdas,
        x_start=0.0,
        x_end=1.0,
        x_step=0.0001,  # 0.1 mm
        output_dir="dlc_dataset",
        file_prefix="cantilever_beam_dlc_modes1to10_step0p1mm.csv",
):
    """
    Generate a single DLC CSV for the whole beam with high spatial resolution.

    Output columns:
        x, DLC_mode1..DLC_modeN
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate clean x values (0.0000, 0.0001, ..., 1.0000)
    x, x_decimals = generate_positions(x_start, x_end, x_step)

    # Compute DLC DataFrame
    df = compute_dlc_dataframe(x, lambdas)

    # Keep x visually clean; keep DLC in scientific notation
    df["x"] = df["x"].map(lambda v: f"{v:.{x_decimals}f}")

    # Inline step-size tag (no helper needed)
    step_str = f"{x_step:.{x_decimals}f}".replace(".", "p")
    out_path = output_dir / f"{file_prefix}_step_{step_str}.csv"

    df.to_csv(out_path, index=False, float_format="%.12e")

    print(f"Saved DLC CSV to: {out_path}")
    print(f"Rows: {len(df)}  |  Step: {x_step} m  |  Modes: {len(lambdas)}")

    return str(out_path)


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    # Eigenvalues for the first 10 vibration modes of a cantilever beam
    lambdas = np.array([
        1.875104, 4.694091, 7.854757, 10.99554, 14.137168,
        17.278759, 20.420352, 23.561944, 26.70353756, 29.84513021
    ])

    severities = np.array([
        3.738699e-05, 1.481688e-04, 3.308325e-04, 5.838736e-04,
        9.067953e-04, 1.300109e-03, 1.765334e-03, 2.304998e-03,
        2.922637e-03, 3.622794e-03, 4.411021e-03, 5.293878e-03,
        6.278932e-03, 7.374760e-03, 8.590945e-03, 9.938079e-03,
        1.142776e-02, 1.307260e-02, 1.488622e-02, 1.688323e-02
    ])

    # --------------------------------------------------------
    # Combined parameter C (user-defined)
    #
    # C should be:
    #   C = sqrt(EI/(rho*A)) / (2*pi*L^2)
    #
    # Provide C directly here (example placeholder value):
    # --------------------------------------------------------
    C = 1.15  # <-- REPLACE with your computed/known C

    # Generate RFS + damaged frequency + delta frequency CSVs
    f_healthy = generate_rfs_and_frequency_csvs(
        lambdas=lambdas,
        severities=severities,
        C=C,
        x_start=0.0,
        x_end=1.0,
        x_step=0.002,
        file_prefix="cantilever_beam_raw_rfs_modes1to10",
        output_dir="single_crack_csvs",
        fdam_output_dir="single_crack_fdam_csvs",
        deltaf_output_dir="single_crack_deltaf_csvs",
        fdam_prefix="cantilever_beam_fdam_modes1to10",
        deltaf_prefix="cantilever_beam_deltaf_modes1to10",
    )

    # --------------------------------------------------------
    # Generate DLC dataset with step size 0.1 mm
    # --------------------------------------------------------
    generate_dlc_csv(
        lambdas=lambdas,
        x_start=0.0,
        x_end=1.0,
        x_step=0.0001,  # 0.1 mm
        output_dir="dlc_dataset",
        file_prefix="cantilever_beam_dlc_modes1to10",
    )
