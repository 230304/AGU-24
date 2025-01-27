def calculate_missing_indices(start_year, end_year, missing_years):
    """
    Calculate the missing indices for each month, given the start year, end year, and missing years.

    Parameters:
    ----------
    start_year : int
        The first year of the dataset.
    end_year : int
        The last year of the dataset.
    missing_years : dict
        A dictionary where keys are months (1-12) and values are the years with missing data.

    Returns:
    -------
    dict
        A dictionary where keys are months (1-12) and values are the missing indices.
    
    Raises:
    ------
    ValueError
        If a year in `missing_years` is outside the range of `start_year` to `end_year`.
    """
    # Initialize a dictionary to hold the missing indices for each month
    missing_indices = {}

    # Calculate the total number of years in the dataset
    total_years = end_year - start_year + 1

    for month, missing_year in missing_years.items():
        # Calculate the index of the missing year relative to `start_year`
        missing_index = missing_year - start_year

        # Ensure the missing index is within the valid range
        if 0 <= missing_index < total_years:
            missing_indices[month] = missing_index
        else:
            raise ValueError(
                f"Missing year {missing_year} for month {month} is out of the data range {start_year}-{end_year}"
            )

    return missing_indices


# Example usage
if __name__ == "__main__":
    # Define the start and end years of the dataset
    start_year = 2003
    end_year = 2021

    # Define missing years for certain months
    missing_years = {
        1: 2018, 2: 2018, 3: 2018, 4: 2018, 5: 2018,  # Missing 2018 data for months 1 to 5
        7: 2017, 8: 2017, 9: 2017, 10: 2017, 11: 2017, 12: 2017  # Missing 2017 data for months 7 to 12
    }

    # Calculate the missing indices
    try:
        missing_indices = calculate_missing_indices(start_year, end_year, missing_years)
        print("Missing indices:", missing_indices)
    except ValueError as e:
        print(f"Error: {e}")
