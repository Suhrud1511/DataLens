from helpers import *
import pandas as pd
import numpy as np

def perform_eda(df, is_preprocessed=False):
    """
    Perform Exploratory Data Analysis on the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        is_preprocessed (bool): Flag indicating if the data is already preprocessed.
    
    Returns:
        tuple: A tuple containing the EDA results dictionary and the preprocessed DataFrame.
    """
    
    print("Starting Exploratory Data Analysis.")
    eda_results = {}

    eda_results['missing_values'] = df.isnull().sum().any()
    print(f"Missing values present: {eda_results['missing_values']}")

    numerical_columns, categorical_columns = identify_columns(df)
    eda_results['numerical_columns'] = numerical_columns
    eda_results['categorical_columns'] = categorical_columns
    
    print(f"Numerical columns: {numerical_columns}")
    print(f"Categorical columns: {categorical_columns}")

    eda_results['number_of_unique_values'] = analyze_categorical(df, categorical_columns)
    eda_results['numerical_columns_analysis'] = analyze_numerical(df, numerical_columns)

    if not is_preprocessed:
        eda_results['scaling_techniques'] = determine_scaling_technique(df, numerical_columns)
    
    eda_results['correlation_matrix'] = correlation_analysis(df, numerical_columns)
    eda_results['distribution_plots'] = distribution_analysis(df, numerical_columns)
    eda_results['boxplot_plots'] = boxplot_analysis(df, numerical_columns)
    eda_results['pairplot_plot'] = pairplot_analysis(df, numerical_columns)
    eda_results['missing_values_heatmap'] = missing_values_heatmap(df)
    eda_results['outlier_plots'] = outlier_analysis(df, numerical_columns)
    eda_results['kde_plots'] = kde_analysis(df, numerical_columns)
    eda_results['violin_plots'] = violin_plot_analysis(df, numerical_columns)
    eda_results['joint_plots'] = joint_plot_analysis(df, numerical_columns)
    eda_results['heatmap_image'] = heatmap_analysis(eda_results['correlation_matrix'])

    if not is_preprocessed:
        preprocessed_df = preprocess(df, eda_results)
    else:
        preprocessed_df = df

    if 'target' in df.columns and not is_preprocessed:
        eda_results['feature_importance'] = feature_importance_analysis(df, 'target')

    if 'date' in df.columns and any(df.dtypes == 'float64'):
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        eda_results['time_series'] = time_series_analysis(df, 'date', numeric_col)

    eda_results['dimensionality_reduction'] = dimensionality_reduction(df, eda_results['numerical_columns'])
    eda_results['anomaly_detection'] = anomaly_detection(df, eda_results['numerical_columns'])
    eda_results['statistical_tests'] = statistical_tests(df, eda_results['numerical_columns'], eda_results['categorical_columns'])

    if any(df.dtypes == 'object') and not is_preprocessed:
        text_col = df.select_dtypes(include=['object']).columns[0]
        eda_results['wordcloud'], eda_results['sentiment_analysis'] = text_data_analysis(df, text_col)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        eda_results['geospatial'] = geospatial_analysis(df, 'latitude', 'longitude')

    eda_results['data_quality_score'] = data_quality_score(df)
    eda_results['automated_insights'] = generate_automated_insights(df, eda_results)

    print("Exploratory Data Analysis complete.")
    return eda_results, preprocessed_df





# Main execution
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\suhru\OneDrive\Desktop\Github\DataLens\backend\data\possum.csv', encoding='latin1')
    
    eda_results, preprocessed_df = perform_eda(df)
    
    # Generate report for original data
    generate_html_report(df, eda_results, preprocessed_df, output_file='Original_EDA_Report.html')
    
    # Perform EDA on preprocessed data
    preprocessed_eda_results, _ = perform_eda(preprocessed_df)
    
    # Generate report for preprocessed data
    generate_html_report(preprocessed_df, preprocessed_eda_results, preprocessed_df, output_file='Preprocessed_EDA_Report.html')

    print("EDA process completed. Check the 'Original_EDA_Report.html' and 'Preprocessed_EDA_Report.html' files for results.")