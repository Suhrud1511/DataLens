import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
import base64
from io import BytesIO

custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#7FDBFF", "#F012BE", "#01FF70", "#FFD700", "#B10DC9"]

def plot_to_image(fig, file_format='png'):
    for ax in fig.axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        if ax.title:
            ax.title.set_color('white')
    buf = BytesIO()
    fig.savefig(buf, format=file_format, facecolor='black', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/{file_format};base64,{img_str}" />'

plt.style.use('dark_background')
sns.set_style("darkgrid")

def identify_columns(df):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_columns, categorical_columns

def analyze_categorical(df, categorical_columns):
    print(f"Analyzing categorical columns: {categorical_columns}")
    return {col: df[col].nunique() for col in categorical_columns}

def analyze_numerical(df, numerical_columns):
    print(f"Analyzing numerical columns: {numerical_columns}")
    metrics = {}
    for col in numerical_columns:
        metrics[col] = {
            'mean': df[col].mean(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurt()
        }
    return metrics

def determine_scaling_technique(df, numerical_columns):
    print("Determining scaling techniques for numerical columns.")
    feature_scaler = {}
    for col in numerical_columns:
        min_val, max_val = df[col].min(), df[col].max()
        range_val = max_val - min_val
        if range_val > 2000:
            feature_scaler[col] = StandardScaler()
        elif min_val >= 0:
            feature_scaler[col] = MinMaxScaler()
        else:
            feature_scaler[col] = RobustScaler()
    return feature_scaler

def correlation_analysis(df, numerical_columns):
    print("Performing correlation analysis.")
    correlation_matrix = df[numerical_columns].corr()
    return correlation_matrix

def distribution_analysis(df, numerical_columns):
    print("Generating distribution plots.")
    distribution_plots = []
    for i, col in enumerate(numerical_columns):
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color=custom_palette[i % len(custom_palette)])
        ax.set_title(f'Distribution of {col}', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        distribution_plots.append(plot_to_image(fig))
        plt.close(fig)
    return distribution_plots

def boxplot_analysis(df, numerical_columns):
    print("Generating boxplot analysis.")
    boxplot_plots = []
    for i, col in enumerate(numerical_columns):
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax, color=custom_palette[i % len(custom_palette)])
        ax.set_title(f'Box Plot of {col}', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        boxplot_plots.append(plot_to_image(fig))
        plt.close(fig)
    return boxplot_plots

def heatmap_analysis(correlation_matrix):
    print("Generating heatmap for correlation matrix.")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix Heatmap', color='white')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    heatmap_image = plot_to_image(fig)
    plt.close(fig)
    return heatmap_image

def pairplot_analysis(df, numerical_columns):
    print("Generating pairplot analysis.")
    sns.set_palette(custom_palette)
    pairplot = sns.pairplot(df[numerical_columns], diag_kind="kde", plot_kws={"s": 5})
    pairplot.fig.suptitle("Pair Plot", y=1.02, color='white')
    pairplot.fig.patch.set_facecolor('black')
    for ax in pairplot.axes.flatten():
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    pairplot_image = plot_to_image(pairplot.fig)
    plt.close(pairplot.fig)
    return pairplot_image

def missing_values_heatmap(df):
    print("Generating missing values heatmap.")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap', color='white')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    missing_values_image = plot_to_image(fig)
    plt.close(fig)
    return missing_values_image

def outlier_analysis(df, numerical_columns):
    print("Performing outlier analysis.")
    outlier_plots = []
    for i, col in enumerate(numerical_columns):
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax, color=custom_palette[i % len(custom_palette)])
        ax.set_title(f'Outlier Analysis of {col}', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        outlier_plots.append(plot_to_image(fig))
        plt.close(fig)
    return outlier_plots

def kde_analysis(df, numerical_columns):
    print("Generating KDE plots.")
    kde_plots = []
    for i, col in enumerate(numerical_columns):
        fig, ax = plt.subplots()
        sns.kdeplot(df[col], ax=ax, color=custom_palette[i % len(custom_palette)])
        ax.set_title(f'KDE Analysis of {col}', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        kde_plots.append(plot_to_image(fig))
        plt.close(fig)
    return kde_plots

def violin_plot_analysis(df, numerical_columns):
    print("Generating violin plots.")
    violin_plots = []
    for i, col in enumerate(numerical_columns):
        fig, ax = plt.subplots()
        sns.violinplot(y=df[col], ax=ax, color=custom_palette[i % len(custom_palette)])
        ax.set_title(f'Violin Plot of {col}', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        violin_plots.append(plot_to_image(fig))
        plt.close(fig)
    return violin_plots

def joint_plot_analysis(df, numerical_columns):
    print("Generating joint plots.")
    joint_plots = []
    for i, col in enumerate(numerical_columns):
        fig = sns.jointplot(x=df.index, y=df[col], kind='scatter', color=custom_palette[i % len(custom_palette)])
        fig.fig.suptitle(f'Joint Plot of {col}', y=1.02, color='white')
        fig.fig.patch.set_facecolor('black')
        fig.ax_joint.tick_params(colors='white')
        fig.ax_joint.xaxis.label.set_color('white')
        fig.ax_joint.yaxis.label.set_color('white')
        fig.ax_marg_x.tick_params(colors='white')
        fig.ax_marg_y.tick_params(colors='white')
        for ax in fig.fig.axes:
            ax.set_facecolor('black')
        joint_plots.append(plot_to_image(fig.fig))
        plt.close(fig.fig)
    return joint_plots

# The rest of your code (perform_eda, preprocess, generate_html_report) remains the same

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

    print("Exploratory Data Analysis complete.")
    return eda_results, preprocessed_df


def preprocess(df, eda_results):
    print("Starting data preprocessing.")
    # Handle missing values
    if eda_results['missing_values']:
        imputer_mean = SimpleImputer(strategy='mean')
        columns_to_impute_mean = [col for col in eda_results['numerical_columns'] if df[col].nunique() > 10]
        df[columns_to_impute_mean] = imputer_mean.fit_transform(df[columns_to_impute_mean])

        imputer_mode = SimpleImputer(strategy='most_frequent')
        columns_to_impute_mode = [col for col in df.columns if col not in columns_to_impute_mean]
        df[columns_to_impute_mode] = imputer_mode.fit_transform(df[columns_to_impute_mode])

        print(f"Missing values handled. Mean imputation for columns: {columns_to_impute_mean} and mode imputation for columns: {columns_to_impute_mode}")

    # Feature scaling
    for col, scaler in eda_results['scaling_techniques'].items():
        df[col] = scaler.fit_transform(df[[col]])
        print(f"Applied {scaler.__class__.__name__} to {col}")

    # Encoding categorical columns
    for col in eda_results['categorical_columns']:
        if df[col].nunique() < 10:
            df[col] = LabelEncoder().fit_transform(df[col])
            print(f"Encoded categorical column: {col}")

    print("Data preprocessing complete.")
    return df

def generate_html_report(df, eda_results, preprocessed_df, output_file='EDA_Report.html'):
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
            {''.join(eda_results['distribution_plots'])}
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

    # Preprocessed Data Preview
    html_content.append(f'''
        <section class="section">
            <h2 class="text-2xl font-bold mb-4">Preprocessed Data Preview</h2>
            <table class="table-auto w-full mb-4">
                <thead>
                    <tr>
                        <th class="border px-4 py-2">Column</th>
                        <th class="border px-4 py-2">Preview</th>
                    </tr>
                </thead>
                <tbody>
    ''')
    for col in preprocessed_df.columns:
        html_content.append(f'''
            <tr>
                <td class="border px-4 py-2">{col}</td>
                <td class="border px-4 py-2">{preprocessed_df[col].head().to_list()}</td>
            </tr>
        ''')
    html_content.append('''
                </tbody>
            </table>
        </section>
    ''')

    with open(output_file, 'w') as file:
        file.write(f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EDA Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
                :root {{
                    --bg-color: #000000;
                    --text-color: #ffffff;
                    --section-bg: #1a1a1a;
                    --border-color: #000000;
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
                h1, h2 {{
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
        ''')
    print(f"HTML report saved as {output_file}")

# Usage
df = pd.read_csv('possum.csv')
eda_results, preprocessed_df = perform_eda(df)
generate_html_report(df, eda_results, preprocessed_df, output_file='report.html')