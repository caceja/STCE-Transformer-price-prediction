# Transformer-CNN-positional-mlp (TCE) — fixed & modernized

import os, time, random
import numpy as np
import pandas as pd
import tensorflow as tf

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention, Conv1D, Add,
    GlobalAveragePooling1D, Layer
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
# Positional Encoding
# --------------------
class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg

    def call(self, inputs):
        # inputs: (B, T, d_model)
        T = tf.shape(inputs)[1]
        pos = tf.cast(tf.range(T)[:, tf.newaxis], tf.float32)   # (T,1)
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / self.d_model)
        )                                                       # (d_model/2,)
        angles = pos * div_term                                 # (T, d_model/2)
        pe = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # (T, d_model)
        pe = pe[tf.newaxis, ...]                                # (1, T, d_model)
        return inputs + pe

# --------------------
# Config
# --------------------
WINDOW_SIZE    = 7                 # total window (seq + target)
SEQ_LENGTH     = WINDOW_SIZE - 1   # input timesteps
NUM_FEATURES   = 1
D_MODEL        = 64
N_HEADS        = 4
CONV_FILTERS   = 64
CONV_KERNEL_SZ = 12                # will be capped to SEQ_LENGTH
DROPOUT_RATE   = 0.2
N_LAYERS       = 2
LR             = 1e-3
EPOCHS         = 100
BATCH_SIZE     = 32
VAL_SPLIT      = 0.1
TRAIN_SIZE     = 2826

INPUT_XLSX     = "Cucumber_FillKNN.xlsx"
OUT_DIR        = "AllData"
OUT_XLSX       = os.path.join(OUT_DIR, "Cucumber_TCE_7seq_90day.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# Helper: sliding windows (seq->next)
# --------------------
def to_sliding_windows(data: np.ndarray, window_size: int):
    """
    Returns:
      X: (n_samples, seq_len, n_features)
      y: (n_samples, n_features)  # next-step target
    """
    seq_len = window_size - 1
    X, y = [], []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]        # (window_size, n_features)
        X.append(window[:seq_len])            # first seq_len as input
        y.append(window[seq_len])             # next value as target
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, data.shape[1])
    return X, y

# --------------------
# Model: Transformer (PE + MHA + FFN) -> CNN residual blocks -> MLP -> next-step
# --------------------
def transformer_model_with_cnn(seq_length: int,
                               num_features: int,
                               d_model: int,
                               n_heads: int,
                               conv_filters: int,
                               conv_kernel_size: int,
                               dropout_rate: float,
                               n_layers: int) -> Model:
    inp = Input(shape=(seq_length, num_features))      # (B, T, F)

    # Project features -> d_model so residuals match
    x = Dense(d_model)(inp)                            # (B, T, d_model)

    # Positional Encoding + dropout
    x = PositionalEncoding(d_model)(x)
    x = Dropout(dropout_rate)(x)

    # Transformer encoder blocks (MHA + FFN)
    for _ in range(n_layers):
        attn_out = MultiHeadAttention(
            num_heads=n_heads,
            key_dim=max(1, d_model // n_heads)         # ensures MHA output dim = d_model
        )(x, x)
        x = Add()([x, Dropout(dropout_rate)(attn_out)])
        x = LayerNormalization(epsilon=1e-6)(x)

        ff = Dense(4 * d_model, activation="relu")(x)
        ff = Dropout(dropout_rate)(ff)
        ff = Dense(d_model)(ff)
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Cap kernel to seq_length to avoid giant padding when kernel > T
    k = min(conv_kernel_size, seq_length)

    # CNN residual blocks (project conv back to d_model for safe residual)
    for _ in range(n_layers):
        conv = Conv1D(filters=conv_filters, kernel_size=k, padding="same", activation="relu")(x)
        conv = Dropout(dropout_rate)(conv)
        if conv_filters != d_model:
            conv = Dense(d_model)(conv)  # time-distributed projection
        x = Add()([x, conv])
        x = LayerNormalization(epsilon=1e-6)(x)

    # MLP head for next-step prediction
    x = GlobalAveragePooling1D()(x)        # (B, d_model)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_features)(x)           # (B, 1)

    return Model(inputs=inp, outputs=out, name="TCE_TransformerCNN_MLP")

# --------------------
# Per-sheet processing
# --------------------
def process_sheet(sheet_name: str, df: pd.DataFrame):
    # Dates & index
    df['Tarikh'] = pd.to_datetime(df['Tarikh'])
    df['Tarikh'] = df['Tarikh'].dt.strftime('%d/%m/%Y')
    df.index = df['Tarikh']
    df = df.drop(columns=['Tarikh'])

    # Ensure numeric
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # Target column
    if 'WholesalePriceNew' not in df.columns:
        raise ValueError(f"'WholesalePriceNew' not found in sheet '{sheet_name}'")
    s = df['WholesalePriceNew'].astype(float)

    # Split
    train_vals = s.iloc[:TRAIN_SIZE].to_numpy().reshape(-1, 1)
    test_vals  = s.iloc[TRAIN_SIZE:].to_numpy().reshape(-1, 1)

    if len(train_vals) < WINDOW_SIZE or len(test_vals) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough rows in '{sheet_name}' for WINDOW_SIZE={WINDOW_SIZE}. "
            f"Train len={len(train_vals)}, Test len={len(test_vals)}"
        )

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # Windows (seq->next)
    train_x, train_y = to_sliding_windows(train_scaled, WINDOW_SIZE)
    test_x,  test_y  = to_sliding_windows(test_scaled,  WINDOW_SIZE)

    # Build & train
    model = transformer_model_with_cnn(
        seq_length=SEQ_LENGTH,
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        conv_filters=CONV_FILTERS,
        conv_kernel_size=CONV_KERNEL_SZ,
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
    mape = mean_absolute_percentage_error(test_y_rescaled, y_pred_rescaled) * 100.0

    print(f"\nResults for sheet {sheet_name} with dayPredict={dayPredict}:")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | MAPE%: {mape:.4f}")

    return test_y_rescaled, y_pred_rescaled, rmse, mae, mape

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    start_time = time.time()

    # Warn if kernel > seq_length (it will be capped internally)
    if CONV_KERNEL_SZ > SEQ_LENGTH:
        print(f"[Info] conv_kernel_size={CONV_KERNEL_SZ} > seq_length={SEQ_LENGTH}; "
              f"kernel will be capped to {SEQ_LENGTH} inside the model.")

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
                    "MAPE_pct":  [mape] * len(y_true),
                })
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                err_df = pd.DataFrame({"error": [str(e)]})
                err_df.to_excel(writer, sheet_name=f"{sheet_name}_ERROR", index=False)

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"Total runtime: {h} hours, {m} minutes, {s} seconds")
