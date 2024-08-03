import numpy as np
from scipy import stats, linalg
from sklearn.base import BaseEstimator, TransformerMixin

def preprocess(df, eda_results):
    """
    Perform adaptive preprocessing on the input DataFrame with advanced techniques.
    
    This function applies a series of preprocessing techniques that
    automatically adjust based on the characteristics of the dataset:
    - Advanced missing value imputation (KNN, MICE)
    - Dynamic outlier detection and treatment (Winsorization, Elliptic Envelope)
    - Automatic normalization and scaling
    - Advanced encoding of categorical variables
    - Data-driven distribution handling
    - Dynamic multicollinearity reduction
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to preprocess.
    eda_results : dict
        Dictionary containing EDA results.
    
    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame.
    """
    
    numeric_columns = eda_results['numerical_columns']
    categorical_columns = eda_results['categorical_columns']
    
    df = handle_missing_values_advanced(df, numeric_columns, categorical_columns)
    df = handle_outliers_advanced(df, numeric_columns)
    df = normalize_and_scale(df, numeric_columns)
    df = encode_categorical_advanced(df, categorical_columns)
    df = handle_distribution(df, numeric_columns)
    df = handle_multicollinearity(df)
    
    return df

def handle_missing_values_advanced(df, numeric_columns, categorical_columns):
    """
    Handle missing values using advanced imputation techniques.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    numeric_columns : list
        List of numeric column names.
    categorical_columns : list
        List of categorical column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed missing values.
    """
    print("Handling missing values with advanced techniques...")
    
    df = knn_impute(df, numeric_columns)
    df = mice_impute(df, numeric_columns + categorical_columns)
    
    print("Missing values handled using KNN and MICE imputation.")
    return df

def knn_impute(df, columns, k=5):
    """
    Perform KNN imputation on specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list
        List of column names to impute.
    k : int, optional
        Number of nearest neighbors to use (default is 5).

    Returns
    -------
    pandas.DataFrame
        DataFrame with KNN imputed values.
    """
    for col in columns:
        if df[col].isnull().sum() > 0:
            data = df[columns].values
            mask = np.isnan(data[:, columns.index(col)])
            
            valid_data = data[~np.isnan(data).any(axis=1)]
            distances = np.sum((valid_data[:, np.newaxis, :] - data[mask]) ** 2, axis=2)
            
            k = min(k, len(valid_data))
            nearest_indices = np.argpartition(distances, k, axis=0)[:k]
            
            imputed_values = np.mean(valid_data[nearest_indices, columns.index(col)], axis=0)
            df.loc[mask, col] = imputed_values
    
    return df

def mice_impute(df, columns, max_iterations=10):
    """
    Perform MICE (Multivariate Imputation by Chained Equations) imputation.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list
        List of column names to impute.
    max_iterations : int, optional
        Maximum number of iterations for MICE (default is 10).

    Returns
    -------
    pandas.DataFrame
        DataFrame with MICE imputed values.
    """
    for _ in range(max_iterations):
        for col in columns:
            if df[col].isnull().sum() > 0:
                mask = df[col].isnull()
                
                X = df.drop(columns=[col]).values
                y = df[col].values
                valid_mask = ~np.isnan(y)
                
                beta = linalg.lstsq(X[valid_mask], y[valid_mask])[0]
                
                y_imputed = X[mask].dot(beta)
                df.loc[mask, col] = y_imputed
    
    return df

def handle_outliers_advanced(df, numeric_columns):
    """
    Handle outliers using Winsorization and Z-score methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    numeric_columns : list
        List of numeric column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with handled outliers.
    """
    print("Handling outliers with advanced techniques...")
    
    for col in numeric_columns:
        data = df[col].values
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(data, lower_bound, upper_bound)
        
        z_scores = (data - np.mean(data)) / np.std(data)
        
        threshold = 3
        outlier_mask = np.abs(z_scores) > threshold
        
        median_value = np.median(data[~outlier_mask])
        df.loc[outlier_mask, col] = median_value
    
    print("Outliers handled using Winsorization and Z-score method.")
    return df

def normalize_and_scale(df, numeric_columns):
    """
    Normalize and scale numeric columns using Yeo-Johnson transformation and robust scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    numeric_columns : list
        List of numeric column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized and scaled numeric columns.
    """
    print("Normalizing and scaling...")
    for col in numeric_columns:
        data = df[col].values
        
        transformed_data, lambda_param = stats.yeojohnson(data)
        
        median = np.median(transformed_data)
        q1, q3 = np.percentile(transformed_data, [25, 75])
        iqr = q3 - q1
        if iqr != 0:
            df[col] = (transformed_data - median) / iqr
        else:
            df[col] = transformed_data - median
    
    print("Normalization and scaling completed using Yeo-Johnson transformation and robust scaling.")
    return df

def encode_categorical_advanced(df, categorical_columns):
    """
    Encode categorical variables using advanced techniques based on cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    categorical_columns : list
        List of categorical column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded categorical variables.
    """
    print("Encoding categorical variables with advanced techniques...")
    for col in categorical_columns:
        n_unique = df[col].nunique()
        n_samples = len(df)
        
        if n_unique / n_samples < 0.05:
            df = one_hot_encode(df, col)
        elif n_unique / n_samples < 0.2:
            df = frequency_encode(df, col)
        else:
            df = hash_encode(df, col)
    
    print("Categorical variables encoded using advanced methods.")
    return df

def one_hot_encode(df, col):
    """
    Perform one-hot encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one-hot encoded column.
    """
    unique_values = df[col].unique()
    for value in unique_values:
        df[f'{col}_{value}'] = (df[col] == value).astype(int)
    df.drop(col, axis=1, inplace=True)
    return df

def frequency_encode(df, col):
    """
    Perform frequency encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with frequency encoded column.
    """
    freq_map = df[col].value_counts(normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(freq_map)
    df.drop(col, axis=1, inplace=True)
    return df

def hash_encode(df, col, n_components=8):
    """
    Perform hash encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.
    n_components : int, optional
        Number of hash components to generate (default is 8).

    Returns
    -------
    pandas.DataFrame
        DataFrame with hash encoded column.
    """
    def hash_function(value):
        return hash(str(value)) % (2**32)
    
    hashed_values = df[col].apply(hash_function)
    for i in range(n_components):
        df[f'{col}_hash_{i}'] = (hashed_values & (1 << i)).astype(int)
    df.drop(col, axis=1, inplace=True)
    return df

def james_stein_encode(df, col):
    """
    Perform James-Stein encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with James-Stein encoded column.
    """
    global_mean = df['target'].mean()
    group_means = df.groupby(col)['target'].mean()
    group_counts = df[col].value_counts()
    
    B = 1 / (1 + (group_means.var() / (global_mean * (1 - global_mean) / group_counts.mean())))
    shrinkage = B * group_means + (1 - B) * global_mean
    
    df[f'{col}_encoded'] = df[col].map(shrinkage)
    df.drop(col, axis=1, inplace=True)
    return df

def m_estimator_encode(df, col):
    """
    Perform M-estimator encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with M-estimator encoded column.
    """
    global_median = df['target'].median()
    group_medians = df.groupby(col)['target'].median()
    group_counts = df[col].value_counts()
    
    k = 1.345
    residuals = (group_medians - global_median) / (group_counts ** 0.5)
    weights = np.where(np.abs(residuals) <= k, 1, k / np.abs(residuals))
    
    shrinkage = weights * group_medians + (1 - weights) * global_median
    
    df[f'{col}_encoded'] = df[col].map(shrinkage)
    df.drop(col, axis=1, inplace=True)
    return df

def target_encode_smoothed(df, col):
    """
    Perform smoothed target encoding on a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    col : str
        Name of the column to encode.

    Returns
    -------
    pandas.DataFrame
        DataFrame with smoothed target encoded column.
    """
    global_mean = df['target'].mean()
    group_means = df.groupby(col)['target'].mean()
    group_counts = df[col].value_counts()
    
    min_samples_leaf = 200
    smoothing = 1 / (1 + np.exp(-(group_counts - min_samples_leaf) / 10))
    
    shrinkage = smoothing * group_means + (1 - smoothing) * global_mean
    
    df[f'{col}_encoded'] = df[col].map(shrinkage)
    df.drop(col, axis=1, inplace=True)
    return df

def handle_distribution(df, numeric_columns):
    """
    Handle distribution of numeric columns using Yeo-Johnson and Box-Cox transformations.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    numeric_columns : list
        List of numeric column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed numeric columns.
    """
    print("Handling distribution...")
    for col in numeric_columns:
        data = df[col].values
        
        transformed_data, lambda_param = stats.yeojohnson(data)
        
        if np.all(transformed_data > 0):
            transformed_data, lambda_param = stats.boxcox(transformed_data)
        
        df[col] = transformed_data
    
    print("Distribution handled using adaptive Yeo-Johnson and Box-Cox transformations.")
    return df

def handle_multicollinearity(df):
    """
    Handle multicollinearity by removing highly correlated features.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with reduced multicollinearity.
    """
    print("Handling multicollinearity...")
    corr_matrix = np.abs(np.corrcoef(df.values.T))
    n = corr_matrix.shape[0]
    
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    
    threshold = max(0.9, 1 - (0.05 * np.log(n)))
    
    high_corr = np.where((corr_matrix > threshold) & mask)
    to_drop = []
    
    for i, j in zip(*high_corr):
        if df.columns[j] not in to_drop:
            var_i = np.var(df.iloc[:, i])
            var_j = np.var(df.iloc[:, j])
            to_drop.append(df.columns[j] if var_i > var_j else df.columns[i])
    
    df.drop(columns=to_drop, inplace=True)
    
    print(f"Multicollinearity handled. Dropped columns: {to_drop}")
    return df