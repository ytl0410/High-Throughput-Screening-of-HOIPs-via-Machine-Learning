#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import joblib

from ase.io import read as ase_read
from dscribe.descriptors import SineMatrix

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model


# ----------------------------
# Utils
# ----------------------------
def extract_number(filename: str) -> int:
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else 10**18


def list_cif_files(path: str):
    if os.path.isfile(path):
        if not path.lower().endswith(".cif"):
            raise ValueError(f"Input file is not .cif: {path}")
        return [path]
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".cif")
        ]
        files = sorted(files, key=lambda p: extract_number(os.path.basename(p)))
        if not files:
            raise ValueError(f"No .cif files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def build_sine_matrix_descriptor(n_atoms_max=50):
    return SineMatrix(
        n_atoms_max=n_atoms_max,
        permutation="none",
        sparse=False
    )


def cif_to_descriptor_df(cif_paths, sine_matrix: SineMatrix):
    descriptors = []
    names = []

    for p in cif_paths:
        name = os.path.basename(p)
        try:
            atoms = ase_read(p)
            desc = sine_matrix.create(atoms)   # (n_atoms_max, n_atoms_max)
            descriptors.append(desc.flatten()) # 2500 dims if n_atoms_max=50
            names.append(name)
        except Exception as e:
            print(f"[WARN] 无法处理文件 {name}: {e}")

    if not descriptors:
        raise RuntimeError("No valid CIFs were processed. Please check your CIF files.")

    df = pd.DataFrame(descriptors, index=names)
    return df


def normalize_and_align(df_raw: pd.DataFrame, scaling_params_path: str):
    """
    Replicates your notebook logic:
      scaling_params = joblib.load(...)
      df_min, range_, columns
      df = df.dropna(axis=1)
      df_normalized = (df - df_min) / range_
      df_normalized = df_normalized.dropna(axis=1)
      descriptors_df_noconstant = df_normalized.loc[:, (df_normalized != df_normalized.iloc[0]).any()]
      descriptors_df_noconstant = descriptors_df_noconstant.dropna(axis=1, how='any')
      df_new = descriptors_df_noconstant.copy(); df_new.columns=str
      df_new = df_new.reindex(columns=columns)
    """
    scaling_params = joblib.load(scaling_params_path)
    df_min = scaling_params["min"]
    range_ = scaling_params["range"]
    columns = scaling_params["columns"]

    df = df_raw.dropna(axis=1)

    # Make sure df_min/range_ align by columns if they are Series
    # If they are numpy arrays, pandas will broadcast by position (as in notebook).
    df_normalized = (df - df_min) / range_
    df_normalized = df_normalized.dropna(axis=1)

    # Remove constant columns (based on first row)
    if len(df_normalized) >= 1:
        descriptors_df_noconstant = df_normalized.loc[
            :, (df_normalized != df_normalized.iloc[0]).any()
        ]
    else:
        descriptors_df_noconstant = df_normalized

    descriptors_df_noconstant = descriptors_df_noconstant.dropna(axis=1, how="any")

    df_new = descriptors_df_noconstant.copy()
    df_new.columns = df_new.columns.astype(str)

    # Align to training columns
    df_new = df_new.reindex(columns=columns)

    return df_new


def build_sigma_model(weights_path: str, input_dim=125):
    # same architecture as your notebook
    num_neural_1 = 64
    num_neural_2 = 128
    num_neural_3 = 512
    num_neural_4 = 512

    A1 = Input(shape=(input_dim,), name="A1")

    A2 = Dense(num_neural_1, activation="relu", name="A2")(A1)
    A3 = Dense(num_neural_2, activation="relu", name="A3")(A2)
    A4 = Dense(num_neural_3, activation="relu", name="A4")(A3)
    A5 = Dense(num_neural_4, activation="relu", name="A5")(A4)
    A6 = Dense(4, name="A6")(A5)

    B2 = Dense(num_neural_1, activation="relu", name="B2")(A1)
    B3 = Dense(num_neural_2, activation="relu", name="B3")(B2)
    B4 = Dense(num_neural_3, activation="relu", name="B4")(B3)
    B5 = Dense(num_neural_4, activation="relu", name="B5")(B4)
    B6 = Dense(4, name="B6")(B5)

    C2 = Dense(num_neural_1, activation="relu", name="C2")(A1)
    C3 = Dense(num_neural_2, activation="relu", name="C3")(C2)
    C4 = Dense(num_neural_3, activation="relu", name="C4")(C3)
    C5 = Dense(num_neural_4, activation="relu", name="C5")(C4)
    C6 = Dense(4, name="C6")(C5)

    concat_layer = Concatenate()([A6, B6, C6])

    model_s = Model(inputs=[A1], outputs=concat_layer)
    model_s.compile(loss="mse", optimizer="adam")
    model_s.load_weights(weights_path)
    return model_s


def predict_all(
    X_125: pd.DataFrame,
    X_228: pd.DataFrame,
    model_dir: str,
):
    pred_df = pd.DataFrame(index=X_125.index)

    # E / G / B (keras SavedModel .keras)
    model_E = load_model(os.path.join(model_dir, "E.keras"), compile=False)
    y = model_E.predict(X_125, verbose=0)
    pred_df["Youngs_Modulus_E_GPa_Hill"] = y.mean(axis=1)

    model_G = load_model(os.path.join(model_dir, "G.keras"), compile=False)
    y = model_G.predict(X_125, verbose=0)
    pred_df["Shear_Modulus_G_GPa_Hill"] = y.mean(axis=1)

    model_B = load_model(os.path.join(model_dir, "B.keras"), compile=False)
    y = model_B.predict(X_125, verbose=0)
    pred_df["Bulk_Modulus_B_GPa_Hill"] = y.mean(axis=1)

    # Sigma_max (custom architecture + weights)
    sigma_weights = os.path.join(model_dir, "sigma_max.h5")
    model_s = build_sigma_model(weights_path=sigma_weights, input_dim=125)
    y_all = model_s.predict(X_125, verbose=0)
    y_reshaped = y_all.reshape((-1, 3, 4))
    y_mean = y_reshaped.mean(axis=1)
    pred_df["Sigma_max"] = y_mean[:, 3]

    # 228-dim models
    model_Ev = load_model(os.path.join(model_dir, "Band_Gap.h5"), compile=False)
    y = model_Ev.predict(X_228, verbose=0)
    pred_df["Ev"] = y.mean(axis=1)

    model_density = load_model(os.path.join(model_dir, "Density.h5"), compile=False)
    y = model_density.predict(X_228, verbose=0)
    pred_df["rho"] = y.mean(axis=1)

    model_E_formation = load_model(os.path.join(model_dir, "E_formation.h5"), compile=False)
    y = model_E_formation.predict(X_228, verbose=0)
    pred_df["E_formation"] = y.mean(axis=1)

    return pred_df


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Predict properties from CIF(s) using SineMatrix + trained ML models."
    )
    ap.add_argument(
        "input_path",
        help="Path to a .cif file OR a directory containing .cif files."
    )
    ap.add_argument(
        "--model-dir",
        default=".",
        help="Directory containing models and scaling params (default: current directory)."
    )
    ap.add_argument(
        "--out",
        default="Prediction_result.csv",
        help="Output CSV filename (default: Prediction_result.csv)."
    )
    ap.add_argument(
        "--n-atoms-max",
        type=int,
        default=50,
        help="n_atoms_max for SineMatrix (default: 50)."
    )
    args = ap.parse_args()

    # Files
    scaling_125 = os.path.join(args.model_dir, "scaling_params_108.pkl")
    scaling_228 = os.path.join(args.model_dir, "scaling_params.pkl")

    for p in [scaling_125, scaling_228]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing scaling params: {p}")

    required_models = [
        "E.keras", "G.keras", "B.keras",
        "sigma_max.h5",
        "Band_Gap.h5", "Density.h5", "E_formation.h5"
    ]
    for m in required_models:
        mp = os.path.join(args.model_dir, m)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Missing model file: {mp}")

    # Read CIF(s) -> raw descriptor
    cif_paths = list_cif_files(args.input_path)
    sine_matrix = build_sine_matrix_descriptor(n_atoms_max=args.n_atoms_max)
    df_raw = cif_to_descriptor_df(cif_paths, sine_matrix)

    # Build X_228 and X_125
    X_228 = normalize_and_align(df_raw, scaling_228)
    X_125 = normalize_and_align(df_raw, scaling_125)

    # Predict
    pred_df = predict_all(X_125=X_125, X_228=X_228, model_dir=args.model_dir)

    # Save
    pred_df.to_csv(args.out, index=True)
    print(f"[DONE] Saved: {args.out}")
    print(pred_df)


if __name__ == "__main__":
    main()