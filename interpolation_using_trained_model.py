def interpolate_missing_data_with_lstm(model, monthly_data, missing_indices, sequence_length):
    """
    Interpolates missing data in monthly time series using an LSTM model.

    Parameters:
    model (tf.keras.Model): Trained LSTM model for interpolation.
    monthly_data (dict): Dictionary with months as keys and corresponding 3D numpy arrays as values.
    missing_indices (dict): Dictionary with months as keys and missing index (year) as values.
    sequence_length (int): Length of the input sequences for LSTM.

    Returns:
    dict: Dictionary containing interpolated data for each month.
    """
    interpolated_data = {}
    total_steps = len(missing_indices) * list(monthly_data.values())[0].shape[0] * list(monthly_data.values())[0].shape[1]
    step = 0

    for month, missing_index in missing_indices.items():
        array = monthly_data[month]
        interpolated_array = np.copy(array)

        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                time_series = array[x, y, :]
                
                if np.isnan(time_series[missing_index]):
                    # Prepare input sequence
                    if missing_index < sequence_length:
                        # If the missing index is less than the sequence length
                        input_seq = np.nan_to_num(time_series[:missing_index], nan=-9999.0)
                        input_seq = np.pad(input_seq, (sequence_length - len(input_seq), 0), 'constant', constant_values=0)
                    else:
                        # For normal cases
                        input_seq = np.nan_to_num(time_series[missing_index - sequence_length:missing_index], nan=-9999.0)
                    
                    input_seq = input_seq.reshape((1, sequence_length, 1))
                    
                    # Predict the missing value using the model
                    interpolated_value = model.predict(input_seq, verbose=0)[0, 0]

                    # Store the interpolated value in the array
                    interpolated_array[x, y, missing_index] = interpolated_value

                # Update progress
                step += 1

        print(f"Progress: {step}/{total_steps} interpolations done for month {month}.")
        
        # Store the interpolated data for the current month
        interpolated_data[month] = interpolated_array
    
    return interpolated_data

# Perform interpolation using the LSTM model
interpolated_monthly_india_data = interpolate_missing_data_with_lstm(model, monthly_india_3d_arrays, missing_indices, sequence_length)
