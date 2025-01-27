import rasterio
import numpy as np
import os
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt


def read_tiff_to_array(tiff_path):
    """Reads a single-band TIFF file into a NumPy array."""
    with rasterio.open(tiff_path) as src:
        return src.read(1)  # Read the first band


def get_shape_of_first_tiff(tiff_files, tiff_directory, india_shape):
    """Determines the shape of arrays from the first valid TIFF file."""
    for file in tiff_files:
        file_path = os.path.join(tiff_directory, file)
        try:
            with rasterio.open(file_path) as src:
                india_shape = india_shape.to_crs(src.crs)  # Match CRS
                out_image, _ = mask(src, india_shape.geometry, crop=True, filled=True)
                if out_image.size > 0:
                    return out_image[0].shape
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    raise FileNotFoundError("No valid TIFF files found to determine array shape.")


def create_monthly_3d_arrays_with_mask(
    tiff_files, missing_files, start_year, end_year, tiff_directory, india_shape
):
    """Creates 3D arrays for monthly data with masking for India."""
    try:
        array_shape = get_shape_of_first_tiff(tiff_files, tiff_directory, india_shape)
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize 3D arrays with NaNs
    total_years = end_year - start_year + 1
    monthly_arrays = {
        month: np.full((array_shape[0], array_shape[1], total_years), np.nan)
        for month in range(1, 13)
    }

    # Convert missing files to a set for faster lookup
    missing_files_set = set(missing_files)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filename = f"TWSA_{year}{month:02d}_cm_CSR_0.25_MASCON_LM.tif"
            file_path = os.path.join(tiff_directory, filename)
            year_index = year - start_year  # Calculate the year index

            if filename in missing_files_set:
                print(f"Skipping missing file: {filename}")
                continue  # Skip if the file is missing

            if os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        india_shape = india_shape.to_crs(src.crs)
                        india_geom = [feature["geometry"] for feature in india_shape.__geo_interface__["features"]]
                        out_image, _ = mask(src, india_geom, crop=True)

                        # Extract the array and handle nodata values
                        array = out_image[0]
                        array = np.ma.masked_where(array == src.nodata, array)

                        # Save a plot of the masked data (optional)
                        plt.figure(figsize=(10, 6))
                        plt.imshow(array, cmap="viridis", interpolation="nearest", vmin=-250, vmax=50)
                        plt.colorbar(label="TWSA")
                        plt.title(f"TWSA for India {year}-{month:02d}")
                        plt.xlabel("Longitude")
                        plt.ylabel("Latitude")
                        plot_save_path = os.path.join("dummy/output/path", f"India_TWSA_{year}_{month:02d}.png")
                        plt.savefig(plot_save_path)
                        plt.close()

                        # Store the array in the correct position
                        monthly_arrays[month][:, :, year_index] = array.filled(np.nan)

                except rasterio.errors.RasterioIOError as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue  # Skip if the file cannot be opened

    return monthly_arrays


if __name__ == "__main__":
    # Dummy file paths and variables
    india_shapefile_path = "dummy/path/India_Shapefiles/Indian_States.shp"
    tiff_directory = "dummy/path/GRACE_DATA/TIFFs"
    missing_files = ["dummy_missing_file1.tif", "dummy_missing_file2.tif"]

    # Load shapefile
    india_shape = gpd.read_file(india_shapefile_path)

    # List of TIFF files in the directory
    tiff_files = sorted([file for file in os.listdir(tiff_directory) if file.endswith(".tif")])

    # Create 3D arrays for monthly data
    monthly_india_3d_arrays = create_monthly_3d_arrays_with_mask(
        tiff_files, missing_files, 2003, 2021, tiff_directory, india_shape
    )

    # Print the shapes of the 3D arrays
    if monthly_india_3d_arrays:
        for month in range(1, 13):
            print(f"Shape of the 3D array for month {month:02d}: {monthly_india_3d_arrays[month].shape}")
