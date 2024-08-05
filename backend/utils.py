import logging
import re
from typing import Dict, List, Optional
from helpers import *
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 
import logging
def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        dict: A dictionary containing the EDA results.
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

    if 'target' in df.columns:
        eda_results['feature_importance'] = feature_importance_analysis(df, 'target')

    if 'date' in df.columns and any(df.dtypes == 'float64'):
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        eda_results['time_series'] = time_series_analysis(df, 'date', numeric_col)

    eda_results['dimensionality_reduction'] = dimensionality_reduction(df, eda_results['numerical_columns'])
    eda_results['anomaly_detection'] = anomaly_detection(df, eda_results['numerical_columns'])
    eda_results['statistical_tests'] = statistical_tests(df, eda_results['numerical_columns'], eda_results['categorical_columns'])

    if any(df.dtypes == 'object'):
        text_col = df.select_dtypes(include=['object']).columns[0]
        eda_results['wordcloud'], eda_results['sentiment_analysis'] = text_data_analysis(df, text_col)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        eda_results['geospatial'] = geospatial_analysis(df, 'latitude', 'longitude')

    eda_results['data_quality_score'] = data_quality_score(df)
    eda_results['automated_insights'] = generate_automated_insights(df, eda_results)

    print("Exploratory Data Analysis complete.")
    return eda_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedPreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'handle_imbalance': False,
            'imbalance_method': 'smote',
            'use_robust_scaler': False,
            'dimensionality_reduction': False,
            'pca_components': 0.95,
        }
        self.imputers: Dict = {}
        self.scalers: Dict = {}
        self.encoders: Dict = {}
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

    def preprocess(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Main preprocessing function that applies all necessary steps.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str, optional): Name of the target column, if any

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        try:
            logging.info("Starting preprocessing...")

            self.dynamic_config_determiner(df, target_column)

            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            datetime_columns = df.select_dtypes(include=['datetime64']).columns
            text_columns = self.identify_text_columns(df)

            df = self.handle_missing_values(df, numeric_columns, categorical_columns)
            df = self.handle_outliers(df, numeric_columns)
            df = self.normalize_and_scale(df, numeric_columns)
            df = self.encode_categorical(df, categorical_columns)
            df = self.process_text_data(df, text_columns)
            df = self.handle_time_series(df, datetime_columns)
            df = self.feature_selection(df, target_column)

            if target_column and self.config.get('handle_imbalance', False):
                df = self.handle_class_imbalance(df, target_column)

            if self.config.get('dimensionality_reduction', False):
                df = self.reduce_dimensions(df)

            logging.info("Preprocessing completed successfully.")
            return df

        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise

    def dynamic_config_determiner(self, df: pd.DataFrame, target_column: Optional[str] = None) -> None:
        """
        Dynamically determine and set the configuration based on dataset characteristics.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str, optional): Name of the target column, if any
        """
        logging.info("Determining dynamic configuration...")
        
        if target_column and df[target_column].nunique() == 2:
            class_counts = df[target_column].value_counts()
            self.config['handle_imbalance'] = min(class_counts) / max(class_counts) < 0.5

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_counts = df[numeric_columns].apply(lambda x: ((x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) | (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))).sum())
        self.config['use_robust_scaler'] = (outlier_counts > len(df) * 0.05).any()

        self.config['dimensionality_reduction'] = df.shape[1] > 50

    def identify_text_columns(self, df: pd.DataFrame, threshold: int = 100) -> List[str]:
        """Identify columns that are likely to contain text data."""
        return [col for col in df.select_dtypes(include=['object']) if df[col].str.len().mean() > threshold]

    def handle_missing_values(self, df: pd.DataFrame, numeric_columns: pd.Index, categorical_columns: pd.Index) -> pd.DataFrame:
        logging.info("Handling missing values...")
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                logging.debug(f"Imputing numeric column: {col}")
                logging.debug(f"Shape before imputation: {df[col].shape}")
                self.imputers[col] = KNNImputer(n_neighbors=5)
                imputed_values = self.imputers[col].fit_transform(df[[col]])
                logging.debug(f"Shape after imputation: {imputed_values.shape}")
                if imputed_values.size > 0:
                    df[col] = imputed_values.ravel()
                else:
                    logging.warning(f"Imputation returned no values for column {col}. Keeping original values.")

        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                logging.debug(f"Imputing categorical column: {col}")
                logging.debug(f"Shape before imputation: {df[col].shape}")
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
                imputed_values = self.imputers[col].fit_transform(df[[col]])
                logging.debug(f"Shape after imputation: {imputed_values.shape}")
                if imputed_values.size > 0:
                    df[col] = imputed_values.ravel()
                else:
                    logging.warning(f"Imputation returned no values for column {col}. Keeping original values.")

        return df
    def handle_outliers(self, df: pd.DataFrame, numeric_columns: pd.Index) -> pd.DataFrame:
        logging.info("Handling outliers...")
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df

    def normalize_and_scale(self, df: pd.DataFrame, numeric_columns: pd.Index) -> pd.DataFrame:
        logging.info("Normalizing and scaling numeric data...")
        for col in numeric_columns:
            self.scalers[col] = RobustScaler() if self.config.get('use_robust_scaler', False) else StandardScaler()
            df[col] = self.scalers[col].fit_transform(df[[col]])
        return df

    def encode_categorical(self, df: pd.DataFrame, categorical_columns: pd.Index) -> pd.DataFrame:
        logging.info("Encoding categorical variables...")
        for col in categorical_columns:
            if df[col].nunique() / len(df) < 0.05:  # One-hot encoding for low cardinality
                df = pd.get_dummies(df, columns=[col], prefix=col)
            else:  # Label encoding for high cardinality
                df[col] = df[col].astype('category').cat.codes
        return df

    def process_text_data(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        logging.info("Processing text data...")
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        for col in text_columns:
            df[col] = df[col].apply(lambda x: self.preprocess_text(x, stop_words, lemmatizer))
        return df

    @staticmethod
    def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

    def handle_time_series(self, df: pd.DataFrame, datetime_columns: pd.Index) -> pd.DataFrame:
        logging.info("Handling time series data...")
        for col in datetime_columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        return df

    def feature_selection(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        logging.info("Performing feature selection...")
        selector = VarianceThreshold()
        df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            selector = SelectKBest(f_classif, k=min(50, len(X.columns)))
            X_new = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            df = pd.concat([df[selected_features], df[target_column]], axis=1)

        return df

    def handle_class_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        logging.info("Handling class imbalance...")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if self.config.get('imbalance_method', 'smote') == 'smote':
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
        else:
            rus = RandomUnderSampler()
            X, y = rus.fit_resample(X, y)

        return pd.concat([X, pd.Series(y, name=target_column)], axis=1)

    def reduce_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Performing dimensionality reduction...")
        pca = PCA(n_components=self.config.get('pca_components', 0.95))
        df_pca = pca.fit_transform(df)
        return pd.DataFrame(df_pca, columns=[f'PC_{i+1}' for i in range(df_pca.shape[1])])


