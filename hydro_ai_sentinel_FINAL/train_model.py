"""
train_model.py
===============
Hydro-AI Sentinel — Kandy Otieno Genga | SCT213-C002-0106/2022
JKUAT BSc. Data Science 2026

PURPOSE:
  Build, train, and evaluate the Bi-Directional LSTM flood model.

ARCHITECTURE:
  Input  (30 days × 12 features)
    → Bidirectional LSTM (128 units, return_sequences=True)
    → Dropout (0.2)
    → Bidirectional LSTM (64 units)
    → Dropout (0.2)
    → Dense (32, ReLU)
    → Dense (1)   ← river level prediction in normalised scale

RUN:
  python train_model.py
  (For ~20-min training on GPU, use Google Colab T4)

OUTPUT:
  models/bilstm_nzoia.keras
  models/bilstm_tana.keras
  models/training_history_nzoia.pkl
  models/training_history_tana.pkl
  outputs/evaluation_nzoia.csv
  outputs/evaluation_tana.csv
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

# ── TensorFlow / Keras ───────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ── Reproducibility ──────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Hyperparameters ──────────────────────────────────────────────
EPOCHS        = 60
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
PATIENCE      = 10          # early-stopping patience

FEATURE_COLS = [
    "rainfall_mm", "river_level_m", "temperature_c", "humidity_pct",
    "rain_3d_sum", "rain_7d_sum", "level_lag1", "level_lag3",
    "level_7d_mean", "level_7d_std", "month_sin", "month_cos",
]
TARGET_INDEX = FEATURE_COLS.index("river_level_m")   # = 1


# ── Model factory ────────────────────────────────────────────────
def build_bilstm(seq_len, n_features):
    """
    Bi-Directional LSTM:
      • Processes the 30-day sequence FORWARD and BACKWARD
      • Dropout(0.2) prevents overfitting
      • Dense head converts LSTM output to a single number
    """
    model = Sequential(
        [
            Input(shape=(seq_len, n_features)),

            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),

            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),

            Dense(32, activation="relu"),
            Dense(1),                           # regression output
        ],
        name="BiLSTM_FloodPredictor",
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Inverse-transform helper ─────────────────────────────────────
def to_metres(norm_values, scaler):
    """
    Undo Min-Max scaling for the river_level_m column only.
    We reconstruct a dummy matrix so sklearn's inverse_transform
    works correctly on the right column.
    """
    n_feat  = len(FEATURE_COLS)
    dummy   = np.zeros((len(norm_values), n_feat), dtype=np.float32)
    dummy[:, TARGET_INDEX] = norm_values.ravel()
    return scaler.inverse_transform(dummy)[:, TARGET_INDEX]


# ── Per-basin training ────────────────────────────────────────────
def train_basin(basin):
    bar = "=" * 58
    print(f"\n{bar}\n  Training Bi-LSTM — {basin.upper()}\n{bar}")

    # ── Load sequences ────────────────────────────────────────────
    seq_path = f"data/sequences_{basin}.npz"
    if not os.path.exists(seq_path):
        raise FileNotFoundError(
            f"{seq_path} not found. Run preprocess.py first."
        )

    seq = np.load(seq_path)
    X_train, y_train = seq["X_train"], seq["y_train"]
    X_val,   y_val   = seq["X_val"],   seq["y_val"]
    X_test,  y_test  = seq["X_test"],  seq["y_test"]

    with open(f"models/scaler_{basin}.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]

    print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

    # ── Build model ───────────────────────────────────────────────
    model = build_bilstm(seq_len, n_features)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────
    model_path = f"models/bilstm_{basin}.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────
    t0      = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    n_epochs = len(history.history["loss"])

    print(f"\n  ✅  Finished in {elapsed:.0f}s  ({n_epochs} epochs)")
    print(f"  Best val_loss = {min(history.history['val_loss']):.6f}")

    # ── Save training history ──────────────────────────────────────
    hist_path = f"models/training_history_{basin}.pkl"
    with open(hist_path, "wb") as fh:
        pickle.dump(history.history, fh)
    print(f"  History → {hist_path}")

    # ── Evaluate on test set ──────────────────────────────────────
    y_pred_norm = model.predict(X_test, verbose=0).ravel()

    y_pred_m = to_metres(y_pred_norm, scaler)
    y_true_m = to_metres(y_test,      scaler)

    rmse = float(np.sqrt(np.mean((y_true_m - y_pred_m) ** 2)))
    mae  = float(np.mean(np.abs(y_true_m - y_pred_m)))
    ss_res = np.sum((y_true_m - y_pred_m) ** 2)
    ss_tot = np.sum((y_true_m - y_true_m.mean()) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f"\n  ── Test-set metrics ──────────────────────────────────")
    print(f"  RMSE : {rmse:.4f} m")
    print(f"  MAE  : {mae:.4f} m")
    print(f"  R²   : {r2:.4f}")

    # ── Save evaluation CSV ───────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    eval_df = pd.DataFrame({
        "actual_level_m"    : y_true_m,
        "predicted_level_m" : y_pred_m,
        "error_m"           : y_true_m - y_pred_m,
    })
    eval_path = f"outputs/evaluation_{basin}.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"  Predictions → {eval_path}")

    return {"basin": basin, "rmse": rmse, "mae": mae, "r2": r2,
            "epochs": n_epochs}


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    results = []
    for basin in ["nzoia", "tana"]:
        results.append(train_basin(basin))

    print("\n\n" + "=" * 58)
    print("  FINAL SUMMARY")
    print("=" * 58)
    for r in results:
        print(
            f"  {r['basin'].upper():8s}  "
            f"RMSE={r['rmse']:.4f} m  "
            f"MAE={r['mae']:.4f} m  "
            f"R²={r['r2']:.4f}  "
            f"({r['epochs']} epochs)"
        )

    print("\n✅  Done.  Download models/ and outputs/ folders.")
    print("    Then run in VSCode:  streamlit run app.py")
