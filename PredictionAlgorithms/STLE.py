# Transformer-LSTM (STLE) — fixed & modernized
# - Removes deprecated experimental.preprocessing import
# - Proper residual shapes via Dense(d_model)
# - Seq-to-one forecasting (predict next step)
# - EarlyStopping & validation split
# - Excel output per sheet

import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention, LSTM, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
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
WINDOW_SIZE   = 7           # total window (seq + target)
SEQ_LENGTH    = WINDOW_SIZE - 1  # input timesteps
NUM_FEATURES  = 1
D_MODEL       = 64
N_HEADS       = 4
FF_DIM        = 128
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.2
N_LAYERS      = 2
LR            = 1e-3
EPOCHS        = 100
BATCH_SIZE    = 32
VAL_SPLIT     = 0.1
TRAIN_SIZE    = 2826        # hard-coded like your original
INPUT_XLSX    = "Cucumber_FillKNN.xlsx"
OUT_DIR       = "AllData"
OUT_XLSX      = os.path.join(OUT_DIR, "Cucumber_STLE_7seq_90day.xlsx")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# Helper: sliding windows (seq->next)
# --------------------
def to_sliding_windows(data: np.ndarray, window_size: int):
    """
    Returns X with shape (n_samples, seq_len, n_features)
            y with shape (n_samples, n_features) for next-step prediction.
    """
    seq_len = window_size - 1
    X, y = [], []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]              # (window_size, n_features)
        X.append(window[:seq_len])                  # first seq_len as input
        y.append(window[seq_len])                   # last element as target
    X = np.array(X)                                 # (N, seq_len, n_features)
    y = np.array(y).reshape(-1, data.shape[1])      # (N, n_features)
    return X, y

# --------------------
# Model: Transformer Encoder stack -> LSTM -> Dense
# --------------------
def transformer_lstm(seq_length: int,
                     num_features: int,
                     d_model: int = 64,
                     n_heads: int = 4,
                     ff_dim: int = 128,
                     lstm_units: int = 64,
                     dropout_rate: float = 0.2,
                     n_layers: int = 2) -> Model:
    """
    Projects inputs to d_model, stacks Transformer encoder blocks,
    then an LSTM (last output) and Dense to 1 feature.
    """
    inp = Input(shape=(seq_length, num_features))          # (B, T, F)

    # Project features -> d_model so residuals match
    x = Dense(d_model)(inp)                                # (B, T, d_model)

    for _ in range(n_layers):
        # Multi-Head Self-Attention + residual + norm
        attn_out = MultiHeadAttention(
            num_heads=n_heads,
            key_dim=max(1, d_model // n_heads)  # per-head dim
        )(x, x)
        x = Add()([x, Dropout(dropout_rate)(attn_out)])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward block + residual + norm
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dropout(dropout_rate)(ff)
        ff = Dense(d_model)(ff)
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)

    # LSTM over encoder outputs (use last hidden state)
    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(num_features)(x)  # predict next step
    return Model(inputs=inp, outputs=out, name="STLE")

# --------------------
# Per-sheet processing
# --------------------
def process_sheet(sheet_name: str, df: pd.DataFrame):
    # Parse and set date index
    df['Tarikh'] = pd.to_datetime(df['Tarikh'])
    df['Tarikh'] = df['Tarikh'].dt.strftime('%d/%m/%Y')
    df.index = df['Tarikh']
    df = df.drop(columns=['Tarikh'])

    # Ensure numeric
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # Select the target column
    if 'WholesalePriceNew' not in df.columns:
        raise ValueError(f"'WholesalePriceNew' not found in sheet '{sheet_name}'")
    s = df['WholesalePriceNew'].astype(float)

    # Train/Test split (like original)
    train_vals = s.iloc[:TRAIN_SIZE].to_numpy().reshape(-1, 1)
    test_vals  = s.iloc[TRAIN_SIZE:].to_numpy().reshape(-1, 1)

    # Edge case: ensure both splits have enough samples for windows
    if len(train_vals) < WINDOW_SIZE or len(test_vals) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough rows in sheet '{sheet_name}' for WINDOW_SIZE={WINDOW_SIZE}. "
            f"Train len={len(train_vals)}, Test len={len(test_vals)}"
        )

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # Windows
    train_x, train_y = to_sliding_windows(train_scaled, WINDOW_SIZE)
    test_x,  test_y  = to_sliding_windows(test_scaled,  WINDOW_SIZE)

    # Build & train model
    model = transformer_lstm(
        seq_length=SEQ_LENGTH,
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        ff_dim=FF_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        n_layers=N_LAYERS
    )
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=LR))

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        train_x, train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[es],
        verbose=0
    )

    # Predict next-step for each test window
    y_pred = model.predict(test_x, verbose=0)  # (N_test, 1)
    dayPredict = len(test_y)

    # Inverse scale
    y_pred_rescaled = scaler.inverse_transform(y_pred).reshape(-1, 1)
    test_y_rescaled = scaler.inverse_transform(test_y).reshape(-1, 1)

    # Metrics
    rmse = sqrt(mean_squared_error(test_y_rescaled, y_pred_rescaled))
    mae  = mean_absolute_error(test_y_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(test_y_rescaled, y_pred_rescaled)

    print(f"\nResults for sheet {sheet_name} with dayPredict={dayPredict}:")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | MAPE: {mape:.6f}")

    return test_y_rescaled, y_pred_rescaled, rmse, mae, mape

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    start_time = time.time()

    xls = pd.ExcelFile(INPUT_XLSX)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(INPUT_XLSX, sheet_name=sheet_name)
            try:
                y_true, y_pred, rmse, mae, mape = process_sheet(sheet_name, df)
                result_df = pd.DataFrame({
                    "Actual":    y_true.flatten(),
                    "Predicted": y_pred.flatten(),
                    "RMSE":      [rmse] * len(y_true),
                    "MAE":       [mae]  * len(y_true),
                    "MAPE":      [mape] * len(y_true),
                })
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                # Write the error into the workbook so you see which sheet failed
                err_df = pd.DataFrame({"error": [str(e)]})
                err_df.to_excel(writer, sheet_name=f"{sheet_name}_ERROR", index=False)

    # Timing
    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"Total runtime: {h} hours, {m} minutes, {s} seconds")
