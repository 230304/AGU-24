import numpy as np
import tensorflow as tf

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for time-series regression tasks.

    Parameters:
    ----------
    input_shape : tuple
        Shape of the input data (sequence_length, num_features).

    Returns:
    -------
    tf.keras.Model
        A compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# Build and summarize the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
model = build_lstm_model(input_shape)
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=150,
    validation_split=0.1
)

# Extract loss and metrics
train_loss = history.history['loss'][-1]
train_mae = history.history['mae'][-1]
val_loss = history.history['val_loss'][-1]
val_mae = history.history['val_mae'][-1]

# Assuming max and min values of the dataset (replace with pre-computed values if known)
max_value = np.max(all_sequences)
min_value = np.min(all_sequences)
range_value = max_value - min_value

# Normalize metrics by the maximum value
normalized_loss = train_loss / max_value
normalized_mae = train_mae / max_value
normalized_val_loss = val_loss / max_value
normalized_val_mae = val_mae / max_value

# Convert normalized metrics to percentages
percentage_loss = normalized_loss * 100
percentage_mae = normalized_mae * 100
percentage_val_loss = normalized_val_loss * 100
percentage_val_mae = normalized_val_mae * 100

# Display normalized and percentage metrics
print("\n--- Normalized Metrics (by Max Value) ---")
print(f"Normalized Loss: {normalized_loss:.4f}, Percentage Loss: {percentage_loss:.2f}%")
print(f"Normalized MAE: {normalized_mae:.4f}, Percentage MAE: {percentage_mae:.2f}%")
print(f"Normalized Val Loss: {normalized_val_loss:.4f}, Percentage Val Loss: {percentage_val_loss:.2f}%")
print(f"Normalized Val MAE: {normalized_val_mae:.4f}, Percentage Val MAE: {percentage_val_mae:.2f}%")

# Normalize metrics by the range of the dataset
normalized_loss_by_range = train_loss / range_value
normalized_mae_by_range = train_mae / range_value
normalized_val_loss_by_range = val_loss / range_value
normalized_val_mae_by_range = val_mae / range_value

# Convert normalized metrics (by range) to percentages
percentage_loss_by_range = normalized_loss_by_range * 100
percentage_mae_by_range = normalized_mae_by_range * 100
percentage_val_loss_by_range = normalized_val_loss_by_range * 100
percentage_val_mae_by_range = normalized_val_mae_by_range * 100

# Display normalized and percentage metrics (by range)
print("\n--- Normalized Metrics (by Range) ---")
print(f"Normalized Loss (by Range): {normalized_loss_by_range:.4f}, Percentage Loss (by Range): {percentage_loss_by_range:.2f}%")
print(f"Normalized MAE (by Range): {normalized_mae_by_range:.4f}, Percentage MAE (by Range): {percentage_mae_by_range:.2f}%")
print(f"Normalized Val Loss (by Range): {normalized_val_loss_by_range:.4f}, Percentage Val Loss (by Range): {percentage_val_loss_by_range:.2f}%")
print(f"Normalized Val MAE (by Range): {normalized_val_mae_by_range:.4f}, Percentage Val MAE (by Range): {percentage_val_mae_by_range:.2f}%")
