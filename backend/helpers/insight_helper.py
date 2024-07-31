import numpy as np 
from scipy import stats
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