def generate_html_report(df, eda_results, output_file='EDA_Report.html'):
    """
    Generates an HTML report for Exploratory Data Analysis (EDA) and saves it to a specified file.

    This function creates an HTML report containing various analyses, including missing values,
    numerical columns analysis, distribution plots, boxplots, pairplots, KDE analysis, outlier analysis,
    correlation matrix heatmap, and more. The report includes tables, plots, and visualizations based on
    the provided EDA results and preprocessed DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame before preprocessing.
    eda_results (dict): A dictionary containing EDA results, including plots and analysis summaries.
    preprocessed_df (pd.DataFrame): The DataFrame after preprocessing.
    output_file (str): The name of the output HTML file. Default is 'EDA_Report.html'.

    Returns:
    None

    Raises:
    ValueError: If the provided DataFrame or EDA results dictionary is empty or missing required keys.


    """
    print(f"Generating HTML report and saving to {output_file}.")
    
    html_content = []

    # Missing Values
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Missing Values</h2>
            <p>Missing values are present: {eda_results['missing_values']}</p>
            <p>Missing values summary:</p>
            <table class="table-auto w-full mb-4">
                <thead>
                    <tr>
                        <th class="border px-4 py-2">Column</th>
                        <th class="border px-4 py-2">Missing Values</th>
                    </tr>
                </thead>
                <tbody>
    ''')
    for col in df.columns:
        html_content.append(f'''
            <tr>
                <td class="border px-4 py-2">{col}</td>
                <td class="border px-4 py-2">{df[col].isnull().sum()}</td>
            </tr>
        ''')
    html_content.append('''
                </tbody>
            </table>
            <h3 class="text-xl font-semibold mt-4">Missing Values Heatmap</h3>
            {eda_results['missing_values_heatmap']}
        </section>
    ''')

    # Numerical Columns Analysis
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Numerical Columns Analysis</h2>
            <table class="table-auto w-full mb-4">
                <thead>
                    <tr>
                        <th class="border px-4 py-2">Column</th>
                        <th class="border px-4 py-2">Mean</th>
                        <th class="border px-4 py-2">Skewness</th>
                        <th class="border px-4 py-2">Kurtosis</th>
                    </tr>
                </thead>
                <tbody>
    ''')
    for col, metrics in eda_results['numerical_columns_analysis'].items():
        html_content.append(f'''
            <tr>
                <td class="border px-4 py-2">{col}</td>
                <td class="border px-4 py-2">{metrics['mean']:.2f}</td>
                <td class="border px-4 py-2">{metrics['skewness']:.2f}</td>
                <td class="border px-4 py-2">{metrics['kurtosis']:.2f}</td>
            </tr>
        ''')
    html_content.append('''
                </tbody>
            </table>
    ''')

    # Distribution Plots
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Distribution Plots</h2>
            <div class=" flex flex-wrap gap-4">
                {''.join(eda_results['distribution_plots'])}      
            </div>
        </section>
    ''')

    # Boxplots
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Boxplots</h2>
            {''.join(eda_results['boxplot_plots'])}
        </section>
    ''')

    # Pairplot
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Pairplot</h2>
            {eda_results['pairplot_plot']}
        </section>
    ''')

    # KDE Analysis
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">KDE Analysis</h2>
            {''.join(eda_results['kde_plots'])}
        </section>
    ''')

    # Violin Plots
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Violin Plots</h2>
            {''.join(eda_results['violin_plots'])}
        </section>
    ''')

    # Outlier Analysis
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Outlier Analysis</h2>
            {''.join(eda_results['outlier_plots'])}
        </section>
    ''')

    # KDE Analysis
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">KDE Analysis</h2>
            {''.join(eda_results['kde_plots'])}
        </section>
    ''')

    # Correlation Matrix Heatmap
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Correlation Matrix Heatmap</h2>
            {eda_results['heatmap_image']}
        </section>
    ''')

    # Joint Plot Analysis
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Joint Plots</h2>
            {''.join(eda_results['joint_plots'])}
        </section>
    ''')

    if 'feature_importance' in eda_results:
        html_content.append(f'''
            <section class="section">
                <h2 class="text-2xl font-bold mb-4">Feature Importance Analysis</h2>
                {eda_results['feature_importance']}
            </section>
        ''')
    
    if 'time_series' in eda_results:
        html_content.append(f'''
            <section class="section">
                <h2 class="text-2xl font-bold mb-4">Time Series Analysis</h2>
                {eda_results['time_series']}
            </section>
        ''')
    
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Dimensionality Reduction</h2>
            {eda_results['dimensionality_reduction']}
        </section>
    ''')
    
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Anomaly Detection</h2>
            {eda_results['anomaly_detection']}
        </section>
    ''')
    
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Statistical Tests</h2>
            <pre>{eda_results['statistical_tests']}</pre>
        </section>
    ''')
    
    if 'wordcloud' in eda_results:
        html_content.append(f'''
            <section class="section">
                <h2 class="text-2xl font-bold mb-4">Text Data Analysis</h2>
                <h3 class="text-xl font-semibold mt-4">Word Cloud</h3>
                {eda_results['wordcloud']}
                <h3 class="text-xl font-semibold mt-4">Sentiment Analysis</h3>
                {eda_results['sentiment_analysis']}
            </section>
        ''')
    
    if 'geospatial' in eda_results:
        html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Geospatial Analysis</h2>
            <div id="map" style="width: 100%; height: 600px;">
                {eda_results['geospatial']}
            </div>
        </section>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.js"></script>
    ''')
    
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Data Quality Score</h2>
            <p>The overall data quality score is: {eda_results['data_quality_score']:.2f} out of 100</p>
        </section>
    ''')

    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Automated Insights</h2>
            <pre>{eda_results['automated_insights']}</pre>
        </section>
    ''')

    full_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EDA Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            :root {{
                --bg-color: #000000;
                --text-color: #ffffff;
                --section-bg: #1a1a1a;
                --border-color: #333333;
                --highlight-color: #9cf890;
            }}
            body {{
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                margin: 0;
                padding: 0;
                background-color: var(--bg-color);
            }}
            .container {{
                width: 90%;
                margin: auto;
                overflow: hidden;
            }}
            .section {{
                background: var(--section-bg);
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
            }}
            h1, h2, h3 {{
                color: var(--highlight-color);
            }}
            h2 {{
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border: 1px solid var(--border-color);
            }}
            th {{
                background-color: var(--highlight-color);
                color: var(--bg-color);
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="section">
                <h1>Exploratory Data Analysis Report</h1>
            </header>
            {''.join(html_content)}
        </div>
    </body>
    </html>
    '''

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(full_html)
    
    print(f"HTML report saved as {output_file}")
    
    return full_html