import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_sequences_for_month(monthly_data, month, missing_time_index, sequence_length):
    """
    Prepare sequences and targets for a specific month, excluding the missing time index.

    Parameters:
    ----------
    monthly_data : dict
        A dictionary where keys are months (1-12) and values are 3D numpy arrays (time-index, x, y).
    month : int
        The month for which sequences and targets are prepared.
    missing_time_index : int
        The time index to exclude while preparing sequences.
    sequence_length : int
        The length of each input sequence.

    Returns:
    -------
    tuple
        A tuple containing two numpy arrays: sequences and targets.
        - sequences: Array of input sequences (n_samples, sequence_length).
        - targets: Array of target values corresponding to sequences (n_samples,).
    """
    sequences = []
    targets = []
    array = monthly_data[month]

    for x in range(array.shape[0]):  # Iterate over spatial x dimension
        for y in range(array.shape[1]):  # Iterate over spatial y dimension
            time_series = array[x, y, :]  # Extract the time series for the current grid cell
            
            for t in range(len(time_series) - sequence_length):
                # Skip sequences that include the missing time index
                if t + sequence_length == missing_time_index:
                    continue
                
                seq = time_series[t:t + sequence_length]
                target = time_series[t + sequence_length]
                
                # Exclude sequences or targets with NaN values
                if not np.isnan(seq).any() and not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)

    return np.array(sequences), np.array(targets)


# Parameters
sequence_length = 6  # Length of each sequence (choose on the basis of where the missing indices are in the array)
all_sequences = []
all_targets = []

# Iterate through each month and prepare sequences
for month, missing_index in missing_indices.items():
    sequences, targets = prepare_sequences_for_month(
        monthly_india_3d_arrays,  # Placeholder for the monthly 3D data dictionary
        month,
        missing_index,
        sequence_length
    )
    all_sequences.append(sequences)
    all_targets.append(targets)

# Concatenate all sequences and targets
all_sequences = np.concatenate(all_sequences, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Normalize the data (optional, uncomment if needed)
# all_sequences = (all_sequences - np.nanmin(all_sequences)) / (np.nanmax(all_sequences) - np.nanmin(all_sequences))
# all_targets = (all_targets - np.nanmin(all_targets)) / (np.nanmax(all_targets) - np.nanmin(all_targets))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_sequences, all_targets, test_size=0.2, random_state=42
)

# Reshape data for LSTM input (LSTM expects input of shape [samples, time steps, features])
X_train = X_train[..., np.newaxis]  # Adding a feature axis
X_test = X_test[..., np.newaxis]

# Output the shapes for verification
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
