import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
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