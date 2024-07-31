from helpers import *
import pandas as pd
import numpy as np

def perform_eda(df):
    print("Starting Exploratory Data Analysis.")
    eda_results = {}

    # Check for missing values
    eda_results['missing_values'] = df.isnull().sum().any()
    print(f"Missing values present: {eda_results['missing_values']}")

    # Identify columns
    numerical_columns, categorical_columns = identify_columns(df)
    eda_results['numerical_columns'] = numerical_columns
    eda_results['categorical_columns'] = categorical_columns
    
    print(f"Numerical columns: {numerical_columns}")
    print(f"Categorical columns: {categorical_columns}")
    
    # Analyze categorical columns
    eda_results['number_of_unique_values'] = analyze_categorical(df, categorical_columns)

    # Analyze numerical columns
    eda_results['numerical_columns_analysis'] = analyze_numerical(df, numerical_columns)

    # Determine scaling techniques
    eda_results['scaling_techniques'] = determine_scaling_technique(df, numerical_columns)

    # Correlation analysis
    eda_results['correlation_matrix'] = correlation_analysis(df, numerical_columns)

    # Distribution analysis
    eda_results['distribution_plots'] = distribution_analysis(df, numerical_columns)

    # Box plot analysis
    eda_results['boxplot_plots'] = boxplot_analysis(df, numerical_columns)

    # Pair plot analysis
    eda_results['pairplot_plot'] = pairplot_analysis(df, numerical_columns)

    # Missing values heatmap
    eda_results['missing_values_heatmap'] = missing_values_heatmap(df)

    # Outlier analysis
    eda_results['outlier_plots'] = outlier_analysis(df, numerical_columns)

    # KDE analysis
    eda_results['kde_plots'] = kde_analysis(df, numerical_columns)

    # Violin plot analysis
    eda_results['violin_plots'] = violin_plot_analysis(df, numerical_columns)

    # Joint plot analysis
    eda_results['joint_plots'] = joint_plot_analysis(df, numerical_columns)

    # Heatmap for correlation matrix
    eda_results['heatmap_image'] = heatmap_analysis(eda_results['correlation_matrix'])

    # Ensure preprocessed data is returned
    preprocessed_df = preprocess(df, eda_results)

    if 'target' in df.columns:  # Assuming 'target' is the target column
        eda_results['feature_importance'] = feature_importance_analysis(df, 'target')
    
    if 'date' in df.columns and any(df.dtypes == 'float64'):  # Assuming 'date' is the date column
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        eda_results['time_series'] = time_series_analysis(df, 'date', numeric_col)
    
    eda_results['dimensionality_reduction'] = dimensionality_reduction(df, eda_results['numerical_columns'])
    eda_results['anomaly_detection'] = anomaly_detection(df, eda_results['numerical_columns'])
    eda_results['statistical_tests'] = statistical_tests(df, eda_results['numerical_columns'], eda_results['categorical_columns'])
    
    if any(df.dtypes == 'object'):  # Assuming text data is stored as object type
        text_col = df.select_dtypes(include=['object']).columns[0]
        eda_results['wordcloud'], eda_results['sentiment_analysis'] = text_data_analysis(df, text_col)
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        eda_results['geospatial'] = geospatial_analysis(df, 'latitude', 'longitude')
    
    eda_results['data_quality_score'] = data_quality_score(df)
    eda_results['automated_insights'] = generate_automated_insights(df, eda_results)
    
    print("Exploratory Data Analysis complete.")
    return eda_results, df





# Main execution
if __name__ == "__main__":
   
    df = pd.read_csv(r'C:\Users\suhru\OneDrive\Desktop\Github\DataLens\backend\data\possum.csv', encoding='latin1')

    
    eda_results, preprocessed_df = perform_eda(df)

    
    generate_html_report(df, eda_results, preprocessed_df, output_file='Enhanced_EDA_Report.html')

    print("EDA process completed. Check the 'Enhanced_EDA_Report.html' file for results.")