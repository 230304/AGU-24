import os
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt

# Placeholder for user-defined inputs
# Update with your actual paths or logic
global_tws_file = 'path/to/global_tws_file.tif'  # Example global TWS file
gws_anomaly_file = 'path/to/gws_anomaly_file.tif'  # Input GWS anomaly file
india_geom = None  # Replace with actual geometry for clipping
output_directory = 'path/to/output_directory'  # Directory to save interpolated files
missing_indices = {}  # Dictionary with missing month-year mapping
interpolated_monthly_india_data = None  # 3D array with interpolated data

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to generate longitude and latitude arrays
def generate_lon_lat_arrays(transform, shape):
    """Generate longitude and latitude grids from transform and shape."""
    nrows, ncols = shape
    lon = np.zeros((nrows, ncols))
    lat = np.zeros((nrows, ncols))
    for row in range(nrows):
        for col in range(ncols):
            lon[row, col], lat[row, col] = rasterio.transform.xy(transform, row, col)
    return lon, lat

# Function to save interpolated data as a TIFF file
def save_interpolated_tiff(output_path, array, transform, crs):
    """Save a 2D array as a GeoTIFF file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)
    print(f"Saved interpolated TIFF: {output_path}")

# Main logic to process and save interpolated data
def process_and_save_interpolated_data():
    """Process and save interpolated data for missing months and years."""
    if interpolated_monthly_india_data is None or not missing_indices:
        raise ValueError("Missing required data: interpolated_monthly_india_data or missing_indices.")

    # Retrieve CRS and transform from the global TWS file
    with rasterio.open(global_tws_file) as src:
        global_transform = src.transform
        global_crs = src.crs

    # Loop through missing months and indices
    for month, missing_index in missing_indices.items():
        interpolated_array = interpolated_monthly_india_data[month]
        interpolated_2d = interpolated_array[:, :, missing_index]
        
        # Replace nodata values with NaN
        interpolated_2d_masked = np.array(interpolated_2d)
        interpolated_2d_masked[interpolated_2d_masked == -12417.8330078125] = np.nan
        
        # Define output file path
        year = 2003 + missing_index
        output_path = os.path.join(output_directory, f"India_TWSA_{year}_{month:02d}.tif")
        
        # Save as GeoTIFF
        save_interpolated_tiff(output_path, interpolated_2d_masked, global_transform, global_crs)
        
        # Generate and display longitude-latitude grids
        lon, lat = generate_lon_lat_arrays(global_transform, interpolated_2d_masked.shape)
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]  # Set plot extent

        # Plot the interpolated data
        plt.figure(figsize=(10, 6))
        plt.imshow(interpolated_2d_masked, cmap='viridis', interpolation='nearest', extent=extent, vmin=-250, vmax=30)
        plt.colorbar(label='TWSA (cm)')
        plt.title(f'Predicted TWSA for India: {year}-{month:02d}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # Uncomment to save plots as PNG
        # plt.savefig(os.path.join(output_directory, f"India_TWSA_{year}_{month:02d}.png"))
        plt.show()

# Example execution
try:
    process_and_save_interpolated_data()
except Exception as e:
    print(f"An error occurred: {e}")
