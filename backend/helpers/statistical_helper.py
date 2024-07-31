import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#7FDBFF", "#F012BE", "#01FF70", "#FFD700", "#B10DC9"]
from .plotting_helper import plot_to_image


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







