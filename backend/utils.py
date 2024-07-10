import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from wordcloud import WordCloud
from textblob import TextBlob
import geopandas as gpd
import folium
from sklearn.cluster import DBSCAN
import base64
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
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
        if col.lower() in ['latitude', 'longitude']:
            continue  # Skip latitude and longitude
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
    try:
        correlation_matrix = df[numerical_columns].corr()
        return correlation_matrix
    except Exception as e:
        print(f"Error performing correlation analysis: {e}")
        return None

def distribution_analysis(df, numerical_columns):
    print("Generating distribution plots.")
    distribution_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error generating distribution plot for {col}: {e}")
    return distribution_plots

def boxplot_analysis(df, numerical_columns):
    print("Generating boxplot analysis.")
    boxplot_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error generating boxplot for {col}: {e}")
    return boxplot_plots

def heatmap_analysis(correlation_matrix):
    print("Generating heatmap for correlation matrix.")
    try:
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
    except Exception as e:
        print(f"Error generating heatmap for correlation matrix: {e}")
        return None

def pairplot_analysis(df, numerical_columns):
    print("Generating pairplot analysis.")
    try:
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
    except Exception as e:
        print(f"Error generating pairplot: {e}")
        return None

def missing_values_heatmap(df):
    print("Generating missing values heatmap.")
    try:
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
    except Exception as e:
        print(f"Error generating missing values heatmap: {e}")
        return None

def outlier_analysis(df, numerical_columns):
    print("Performing outlier analysis.")
    outlier_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error performing outlier analysis for {col}: {e}")
    return outlier_plots

def kde_analysis(df, numerical_columns):
    print("Generating KDE plots.")
    kde_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error generating KDE plot for {col}: {e}")
    return kde_plots

def violin_plot_analysis(df, numerical_columns):
    print("Generating violin plots.")
    violin_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error generating violin plot for {col}: {e}")
    return violin_plots

def joint_plot_analysis(df, numerical_columns):
    print("Generating joint plots.")
    joint_plots = []
    for i, col in enumerate(numerical_columns):
        try:
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
        except Exception as e:
            print(f"Error generating joint plot for {col}: {e}")
    return joint_plots

def feature_importance_analysis(df, target_column):
    print("Performing feature importance analysis.")
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Random Forest Feature Importance
        features = X.columns
        forest_importances = pd.Series(importances, index=features).sort_values(ascending=False)
        sns.barplot(x=forest_importances, y=forest_importances.index, ax=ax1)
        ax1.set_title("Random Forest Feature Importance")
        ax1.set_xlabel("Importance")
        
        # Mutual Information
        mi_scores = pd.Series(mi_scores, index=features).sort_values(ascending=False)
        sns.barplot(x=mi_scores, y=mi_scores.index, ax=ax2)
        ax2.set_title("Mutual Information Scores")
        ax2.set_xlabel("Mutual Information")
        
        plt.tight_layout()
        feature_importance_plot = plot_to_image(fig)
        plt.close(fig)
        
        return feature_importance_plot
    except Exception as e:
        print(f"Error performing feature importance analysis: {e}")
        return None

def time_series_analysis(df, date_column, value_column):
    print("Performing time series analysis.")
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df = df.sort_index()
        
        # Decomposition
        decomposition = seasonal_decompose(df[value_column], model='additive', period=1)
        
        # Trend plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(decomposition.trend.index, decomposition.trend, label='Trend', color='blue')
        ax.set_title('Trend Component', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        trend_plot = plot_to_image(fig)
        plt.close(fig)
        
        # Seasonal plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(decomposition.seasonal.index, decomposition.seasonal, label='Seasonality', color='green')
        ax.set_title('Seasonal Component', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        seasonal_plot = plot_to_image(fig)
        plt.close(fig)
        
        # Residual plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(decomposition.resid.index, decomposition.resid, label='Residuals', color='red')
        ax.set_title('Residual Component', color='white')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        residual_plot = plot_to_image(fig)
        plt.close(fig)
        
        return trend_plot, seasonal_plot, residual_plot
    except Exception as e:
        print(f"Error performing time series analysis: {e}")
        return None, None, None

def dimensionality_reduction(df, numerical_columns):
    try:
        print("Performing dimensionality reduction.")
        
        # Check if numerical_columns is empty or not present in df
        if not numerical_columns or not all(col in df.columns for col in numerical_columns):
            raise ValueError("Invalid numerical columns provided.")
        
        # Extract numerical data and scale it
        X = StandardScaler().fit_transform(df[numerical_columns])
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X)
        
        # Create plots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "t-SNE"))
        fig.add_trace(go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers', name='PCA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], mode='markers', name='t-SNE'), row=1, col=2)
        
        fig.update_layout(height=500, title_text="Dimensionality Reduction")
        dim_reduction_plot = fig.to_html(full_html=False)
        
        return dim_reduction_plot
    
    except ValueError as ve:
        print(f"ValueError in dimensionality_reduction: {ve}")
        return None
    except Exception as e:
        print(f"Error in dimensionality_reduction: {e}")
        return None

def anomaly_detection(df, numerical_columns):
    try:
        print("Performing anomaly detection.")
        X = df[numerical_columns]
        
        # Check if numerical_columns is empty or not present in df
        if not numerical_columns or not all(col in df.columns for col in numerical_columns):
            raise ValueError("Invalid numerical columns provided.")
        
        # Perform anomaly detection using Elliptic Envelope
        outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
        outlier_labels = outlier_detector.fit_predict(X)
        
        # Create plot
        fig = go.Figure(data=[go.Scatter(
            x=df.index,
            y=df[numerical_columns[0]],
            mode='markers',
            marker=dict(
                color=outlier_labels,
                colorscale='Viridis',
                line_width=2
            )
        )])
        
        fig.update_layout(title=f"Anomaly Detection using {numerical_columns[0]}")
        anomaly_plot = fig.to_html(full_html=False)
        
        return anomaly_plot
    
    except ValueError as ve:
        print(f"ValueError in anomaly_detection: {ve}")
        return None
    except Exception as e:
        print(f"Error in anomaly_detection: {e}")
        return None

def statistical_tests(df, numerical_columns, categorical_columns):
    try:
        print("Performing statistical tests.")
        results = []
        
        # T-test for numerical columns
        for col in numerical_columns:
            t_stat, p_value = stats.ttest_1samp(df[col], 0)
            results.append(f"T-test for {col}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        
        # Chi-square test for categorical columns
        for col in categorical_columns:
            observed = df[col].value_counts()
            chi2, p_value = stats.chisquare(observed)
            results.append(f"Chi-square test for {col}: chi2 = {chi2:.4f}, p-value = {p_value:.4f}")
        
        return "\n".join(results)
    
    except ValueError as ve:
        print(f"ValueError in statistical_tests: {ve}")
        return None
    except Exception as e:
        print(f"Error in statistical_tests: {e}")
        return None

def text_data_analysis(df, text_column):
    print("Performing text data analysis.")
    text = " ".join(df[text_column].astype(str))
    
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    wordcloud_plot = plot_to_image(fig)
    plt.close(fig)
    
    # Sentiment Analysis
    sentiments = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    fig = px.histogram(sentiments, nbins=50, title="Sentiment Distribution")
    sentiment_plot = fig.to_html(full_html=False)
    
    return wordcloud_plot, sentiment_plot

def geospatial_analysis(df, lat_column, lon_column):
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

def data_quality_score(df):
    print("Calculating data quality score.")
    score = 100
    
    # Check for missing values
    missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    score -= missing_percentage
    
    # Check for outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_percentage = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum() / len(df) * 100
        score -= outlier_percentage / len(df.columns)
    
    # Check for data distribution (assuming normal distribution is ideal)
    for col in df.select_dtypes(include=[np.number]).columns:
        _, p_value = stats.normaltest(df[col].dropna())
        if p_value < 0.05:
            score -= 5 / len(df.columns)
    
    return max(0, min(score, 100))

def generate_automated_insights(df, eda_results):
    print("Generating automated insights.")
    insights = []
    
    # Missing values insight
    missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    insights.append(f"The dataset contains {missing_percentage:.2f}% missing values.")
    
    # Correlation insight
    high_correlations = eda_results['correlation_matrix'][abs(eda_results['correlation_matrix']) > 0.8]
    if not high_correlations.empty:
        insights.append("High correlations (>0.8) found between the following features:")
        for col1, col2 in high_correlations.stack().index:
            if col1 != col2:
                insights.append(f"- {col1} and {col2}: {high_correlations.loc[col1, col2]:.2f}")
    
    # Skewness insight
    for col, metrics in eda_results['numerical_columns_analysis'].items():
        if abs(metrics['skewness']) > 1:
            insights.append(f"The feature '{col}' is highly skewed (skewness: {metrics['skewness']:.2f}).")
    
    # Data quality score insight
    quality_score = data_quality_score(df)
    insights.append(f"The overall data quality score is {quality_score:.2f} out of 100.")
    
    return "\n".join(insights)


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

def preprocess(df, eda_results):
    print("Starting data preprocessing.")
    
    try:
        # Handle missing values
        if eda_results['missing_values']:
            # Mean imputation for numerical columns
            imputer_mean = SimpleImputer(strategy='mean')
            columns_to_impute_mean = [col for col in eda_results['numerical_columns'] if df[col].nunique() > 10]
            df[columns_to_impute_mean] = imputer_mean.fit_transform(df[columns_to_impute_mean])

            # Mode imputation for remaining columns
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
    
    except ValueError as ve:
        print(f"ValueError in preprocess: {ve}")
        return None
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return None

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

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f'''
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
        ''')
    print(f"HTML report saved as {output_file}")

# Main execution
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r'C:\Users\suhru\OneDrive\Desktop\Github\DataLens\backend\data\world_country_and_usa_states_latitude_and_longitude_values.csv', encoding='latin1')

    # Perform EDA
    eda_results, preprocessed_df = perform_eda(df)

    # Generate HTML report
    generate_html_report(df, eda_results, preprocessed_df, output_file='Enhanced_EDA_Report.html')

    print("EDA process completed. Check the 'Enhanced_EDA_Report.html' file for results.")