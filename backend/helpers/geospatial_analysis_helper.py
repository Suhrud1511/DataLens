import pandas as pd
import folium
def geospatial_analysis(df, lat_column, lon_column):
    """
    Performs geospatial analysis on a DataFrame by generating an interactive map.

    This function takes a DataFrame containing latitude and longitude columns,
    calculates the mean coordinates to center the map, and adds markers for each
    point in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing geospatial data.
    lat_column (str): The name of the column containing latitude values.
    lon_column (str): The name of the column containing longitude values.

    Returns:
    str: An HTML string representation of the generated map.

    Raises:
    ValueError: If the specified latitude or longitude column does not exist in the DataFrame.
    ValueError: If the DataFrame is empty.
    """
    if lat_column not in df.columns or lon_column not in df.columns:
        raise ValueError("Specified latitude or longitude column does not exist in the DataFrame.")
    
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    
    print("Performing geospatial analysis.")
    
    # Calculate the center of the map
    map_center = [df[lat_column].mean(), df[lon_column].mean()]
    
    # Create a folium map centered at the mean latitude and longitude
    m = folium.Map(location=map_center, zoom_start=6)
    
    # Add markers to the map
    for _, row in df.iterrows():
        folium.Marker(
            [row[lat_column], row[lon_column]],
            popup=f"Lat: {row[lat_column]}, Lon: {row[lon_column]}"
        ).add_to(m)
    
    # Save the map as an HTML string
    map_html = m._repr_html_()
    
    return map_html
