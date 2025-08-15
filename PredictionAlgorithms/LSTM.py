# LSTMtiyah — fixed & modernized

import os, time, math, random
import numpy as np
import pandas as pd
import tensorflow as tf

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------
# Reproducibility
# --------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# --------------------
# Config
# --------------------
INPUT_XLSX   = r"Cucumber_FillKNN.xlsx"
OUT_DIR      = "AllData"
OUT_XLSX     = os.path.join(OUT_DIR, "Cucumber_LSTM_7seq_90day.xlsx")

TIMESTEPS    = 7             # input sequence length
DAY_PREDICT  = 90            # test size (days at end of series)
EPOCHS       = 100
BATCH_SIZE   = 32
UNITS_1      = 100
UNITS_2      = 100
DROPOUT_RATE = 0.0           # set >0 (e.g., 0.2) if you want regularization
VAL_SPLIT    = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# Sliding windows (seq->next)
# --------------------
def make_windows(series_scaled: np.ndarray, timesteps: int):
    """
    series_scaled: (N, 1) scaled series
    returns X: (M, timesteps, 1), y: (M, 1)
    """
    X, y = [], []
    for i in range(timesteps, len(series_scaled)):
        X.append(series_scaled[i - timesteps:i, 0])
        y.append(series_scaled[i, 0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X.reshape((X.shape[0], timesteps, 1)), y

# --------------------
# Per-sheet processing
# --------------------
def process_sheet(sheet_name: str, df: pd.DataFrame):
    if 'WholesalePriceNew' not in df.columns:
        raise ValueError(f"'WholesalePriceNew' not found in sheet '{sheet_name}'")

    # Target series (as float)
    series = df['WholesalePriceNew'].astype(float).to_numpy().reshape(-1, 1)

    if len(series) <= DAY_PREDICT + TIMESTEPS:
        raise ValueError(
            f"Not enough rows for DAY_PREDICT={DAY_PREDICT} and TIMESTEPS={TIMESTEPS} "
            f"in sheet '{sheet_name}'. Need > {DAY_PREDICT + TIMESTEPS}, have {len(series)}."
        )

    # Train/test split (tail of DAY_PREDICT days is test)
    train_len = len(series) - DAY_PREDICT
    train_vals = series[:train_len]
    test_vals  = series[train_len:]  # unscaled, for inverse & metrics

    # Scale on train only; transform test
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals)
    # For test windows, we need timesteps overlap from train tail
    full_scaled  = scaler.transform(series)  # safe since scaler fit on train
    test_scaled  = full_scaled[train_len - TIMESTEPS:]  # include overlap

    # Windows
    X_train, y_train = make_windows(train_scaled, TIMESTEPS)
    # Build test windows to predict exactly len(test_vals) points
    X_test, _ = make_windows(test_scaled, TIMESTEPS)
    # X_test length equals len(test_vals)
    assert X_test.shape[0] == len(test_vals), "Test windowing mismatch."

    # Model
    model = Sequential([
        LSTM(UNITS_1, return_sequences=True, input_shape=(TIMESTEPS, 1)),
        Dropout(DROPOUT_RATE),
        LSTM(UNITS_2, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        shuffle=False,                 # keep temporal order
        callbacks=[es],
        verbose=0
    )

    # Predict
    pred_scaled = model.predict(X_test, verbose=0)      # (N_test, 1)
    preds = scaler.inverse_transform(pred_scaled)       # back to original scale

    # Metrics vs true test (original scale)
    y_true = test_vals                                  # (N_test, 1)
    rmse = sqrt(mean_squared_error(y_true, preds))
    mae  = mean_absolute_error(y_true, preds)
    mse  = mean_squared_error(y_true, preds)
    mape = mean_absolute_percentage_error(y_true, preds) * 100.0

    print(f"\nResults for sheet '{sheet_name}' with dayPredict={DAY_PREDICT}:")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | MSE: {mse:.6f} | MAPE%: {mape:.4f}")

    return y_true.flatten(), preds.flatten(), rmse, mae, mse, mape

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    start = time.time()

    xls = pd.ExcelFile(INPUT_XLSX)
    with pd.ExcelWriter(OUT_XLSX, engine='xlsxwriter') as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(INPUT_XLSX, sheet_name=sheet_name)
            try:
                actual, pred, rmse, mae, mse, mape = process_sheet(sheet_name, df)
                out_df = pd.DataFrame({
                    "Actual":     actual,
                    "Predictions":pred,
                    "RMSE":       [rmse] * len(pred),
                    "MAE":        [mae]  * len(pred),
                    "MSE":        [mse]  * len(pred),
                    "MAPE_pct":   [mape] * len(pred),
                })
                out_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                pd.DataFrame({"error": [str(e)]}).to_excel(
                    writer, sheet_name=f"{sheet_name}_ERROR", index=False
                )

    h, rem = divmod(int(time.time() - start), 3600)
    m, s   = divmod(rem, 60)
    print(f"Total runtime: {h} hours, {m} minutes, {s} seconds")
