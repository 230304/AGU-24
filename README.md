# LSTM to Fill GRACE Gap

## 📌 Project Overview
This repository focuses on addressing the data gap between GRACE and GRACE-FO satellite observations by utilizing Long Short-Term Memory (LSTM) networks. The goal is to develop a machine learning-based interpolation model that reconstructs missing Total Water Storage Anomaly (TWSA) data using historical trends and available hydrological data. The gap-filled dataset is crucial for continuous hydrological studies, climate analysis, and groundwater assessment.

## 🎯 Achievements
- **Data Preprocessing:** Created structured monthly data arrays for analysis.
- **Gap Identification:** Automatically detected missing months in the GRACE dataset.
- **Model Training:** Implemented and trained an LSTM model to learn temporal dependencies in TWSA data.
- **Interpolation:** Applied the trained model to reconstruct missing values.
- **Visualization:** Generated plots to analyze and compare interpolated data with available records.

## 🛠 How to Use the Repository
The repository contains modular scripts to facilitate easy execution. Follow these steps:

### 1️⃣ **Prepare Data**
Run the script to create structured monthly arrays from raw satellite data:
```bash
python create_monthly_arrays.py
```

### 2️⃣ **Identify Missing Data**
Determine which months are missing in the dataset:
```bash
python calculate_missing_indices_of_monthly_arrays.py
```

### 3️⃣ **Prepare Training and Testing Data**
Generate train-test splits for model training:
```bash
python prepare_train_test_data.py
```

### 4️⃣ **Train the LSTM Model**
Build and train the LSTM model using historical data:
```bash
python build_train_and_calculateloss.py
```

### 5️⃣ **Interpolate Missing Values**
Apply the trained model to estimate missing TWSA values:
```bash
python interpolation_using_trained_model.py
```

### 6️⃣ **Visualize and Save Results**
Plot and store the interpolated data for further analysis:
```bash
python plot_save_interpolated_data.py
```

## 📌 Dependencies
Ensure the following Python libraries are installed:
```bash
pip install numpy pandas tensorflow matplotlib xarray rasterio
```

## 🔍 Future Scope
- Extend the model to regional and global scale applications.
- Incorporate additional geophysical parameters for improved accuracy.
- Compare LSTM-based interpolation with statistical and hybrid approaches.

---
For any questions or contributions, feel free to raise an issue or submit a pull request! 🚀

