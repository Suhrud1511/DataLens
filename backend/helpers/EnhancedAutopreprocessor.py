import numpy as np
import pandas as pd
from scipy import stats,signal
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Union, List, Tuple
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
class EnhancedMathAutopreprocessor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'embedding_dim': 3,
            'tau': 1,
            'lyap_k': 5,
            'box_size': 10,
            'q_order': 2,
            'hurst_lags': 20,
            'pca_threshold': 0.95,
            'vif_threshold': 5,
            'correlation_threshold': 0.9,
            'outlier_threshold': 3,
            'polynomial_degree': 2,
        }
    def skewness(self, data: np.ndarray) -> float:
        n = len(data)
        m2 = np.mean((data - np.mean(data))**2)
        m3 = np.mean((data - np.mean(data))**3)
        g1 = m3 / (m2**1.5)
        return ((n-1)*n)**0.5 / (n-2) * g1

    def kurtosis(self, data: np.ndarray) -> float:
        n = len(data)
        m2 = np.mean((data - np.mean(data))**2)
        m4 = np.mean((data - np.mean(data))**4)
        g2 = m4 / (m2**2) - 3
        return (n-1) / ((n-2)*(n-3)) * ((n+1)*g2 + 6)

    def augmented_dickey_fuller(self, data: np.ndarray, max_lag: int = None) -> Tuple[float, float]:
        def estimate_ar1(x):
            return np.linalg.lstsq(x[:-1].reshape(-1, 1), x[1:], rcond=None)[0][0]

        def select_lag(x, max_lag):
            if max_lag is None:
                max_lag = int(np.ceil(12 * (len(x)/100.)**(1/4.)))
            aic = np.zeros(max_lag+1)
            for lag in range(max_lag+1):
                if lag == 0:
                    aic[lag] = np.log(np.sum(x**2)/len(x))
                else:
                    y = x[lag:]
                    X = np.column_stack([x[lag-1:-1], np.ones(len(y))])
                    for i in range(1, lag):
                        X = np.column_stack([X, x[lag-i-1:-i-1]])
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    e = y - np.dot(X, beta)
                    aic[lag] = np.log(np.sum(e**2)/len(y)) + 2*(lag+1)/len(y)
            return np.argmin(aic)

        x = np.asarray(data)
        lagr = select_lag(np.diff(x), max_lag)
        y = np.diff(x)[lagr:]
        X = np.column_stack([x[lagr:-1], np.ones(len(y))])
        for i in range(lagr):
            X = np.column_stack([X, np.diff(x)[lagr-i-1:-i-1]])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - np.dot(X, beta)
        sigma2 = np.dot(e.T, e) / len(y)
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(np.dot(X.T, X))))
        t = beta[0] / se[0]
        p = stats.t.sf(np.abs(t), len(y)-X.shape[1])*2
        return t, p

    def autocorrelation(self, data: np.ndarray, nlags: int) -> np.ndarray:
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)
        norms = data - mean
        acov = np.correlate(norms, norms, 'full')[n-1:]
        acor = acov / (var * n)
        return acor[:nlags+1]

    def partial_autocorrelation(self, data: np.ndarray, nlags: int) -> np.ndarray:
        pacf = np.zeros(nlags + 1)
        pacf[0] = 1
        ar_coef = np.zeros(nlags)

        for k in range(1, nlags + 1):
            r = self.autocorrelation(data, k)[1:]
            if k == 1:
                ar_coef[0] = r[0]
                pacf[1] = ar_coef[0]
            else:
                ar_coef_prev = ar_coef[:k-1]
                ar_coef[:k] = np.linalg.solve(
                    signal.toeplitz(self.autocorrelation(data, k-1)),
                    r
                )
                pacf[k] = ar_coef[k-1]
                ar_coef[:k-1] = ar_coef_prev - pacf[k] * ar_coef_prev[::-1]

        return pacf

    def breusch_pagan_test(self, data: np.ndarray) -> Tuple[float, float]:
        n = len(data)
        y = data
        X = np.column_stack((np.ones(n), np.arange(n)))
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - np.dot(X, b)
        e2 = e**2
        X2 = np.column_stack((np.ones(n), X[:, 1]))
        b2 = np.linalg.lstsq(X2, e2, rcond=None)[0]
        sse = np.sum((e2 - np.dot(X2, b2))**2)
        ssr = np.sum((np.dot(X2, b2) - np.mean(e2))**2)
        lm = n * (1 - (sse / (sse + ssr)))
        p_value = 1 - stats.chi2.cdf(lm, 1)
        return lm, p_value

    def largest_lyapunov_exponent(self, data: np.ndarray, dim: int, tau: int, k: int) -> float:
        N = len(data)
        Y = np.array([data[i:i+dim*tau:tau] for i in range(N - (dim-1)*tau)])
        D = squareform(pdist(Y))
        np.fill_diagonal(D, np.inf)
        
        L = np.zeros(k)
        for i in range(k):
            j = np.argmin(D, axis=1)
            d0 = D[np.arange(len(D)), j]
            D[np.arange(len(D)), j] = np.inf
            d1 = D[np.arange(len(D)), j]
            L[i] = np.mean(np.log(d1/d0))
        
        return np.polyfit(range(k), L, 1)[0]

    def correlation_dimension(self, data: np.ndarray, dim: int, tau: int, eps: float) -> float:
        N = len(data)
        Y = np.array([data[i:i+dim*tau:tau] for i in range(N - (dim-1)*tau)])
        D = squareform(pdist(Y))
        C = np.mean(D < eps)
        return np.log(C) / np.log(eps)

    def multifractal_spectrum(self, data: np.ndarray, q_range: np.ndarray, box_sizes: np.ndarray) -> Dict:
        N = len(data)
        F_q = np.zeros((len(q_range), len(box_sizes)))
        
        for i, q in enumerate(q_range):
            for j, box_size in enumerate(box_sizes):
                n_boxes = N // box_size
                boxes = data[:n_boxes*box_size].reshape(n_boxes, box_size)
                box_sums = np.sum(np.abs(boxes), axis=1)
                F_q[i, j] = np.mean(box_sums**q)**(1/q) if q != 0 else np.exp(np.mean(np.log(box_sums)))
        
        h = np.array([np.polyfit(np.log(box_sizes), np.log(F_q[i]), 1)[0] for i in range(len(q_range))])
        tau = h * q_range - 1
        alpha = np.gradient(tau, q_range)
        f_alpha = q_range * alpha - tau
        
        return {'alpha': alpha, 'f_alpha': f_alpha}

    def hurst_exponent(self, data: np.ndarray, max_lag: int) -> float:
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def sample_entropy(self, data: np.ndarray, m: int, r: float) -> float:
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
            return sum(C)
        
        N = len(data)
        return -np.log(_phi(m+1) / _phi(m))

    @staticmethod
    def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        z_scores = 0.6745 * (data - median) / mad
        return np.abs(z_scores) > threshold

    @staticmethod
    def impute_missing_values(data: np.ndarray) -> np.ndarray:
        is_nan = np.isnan(data)
        if np.any(is_nan):
            non_nan = data[~is_nan]
            data[is_nan] = np.median(non_nan)
        return data

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std if std != 0 else np.zeros_like(data)

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data)

    @staticmethod
    def pca(data: np.ndarray, n_components: int) -> np.ndarray:
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        components = eigenvectors[:, :n_components]
        return np.dot(data_centered, components)

    @staticmethod
    def remove_trend(data: np.ndarray) -> np.ndarray:
        n = len(data)
        X = np.column_stack((np.ones(n), np.arange(n)))
        beta = np.linalg.lstsq(X, data, rcond=None)[0]
        trend = np.dot(X, beta)
        return data - trend

    @staticmethod
    def seasonal_decomposition(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(data)
        trend = np.convolve(data, np.ones(period)/period, mode='same')
        s = np.zeros(period)
        for i in range(period):
            s[i] = np.mean(data[i::period] - trend[i::period])
        s -= np.mean(s)
        seasonal = np.tile(s, n//period + 1)[:n]
        residual = data - trend - seasonal
        return trend, seasonal, residual
    def detect_distribution(self, data: np.ndarray) -> str:
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        # Check for discrete distributions
        if np.all(np.equal(np.mod(data, 1), 0)):
            # Poisson test
            lambda_mle = np.mean(data)
            if self.poisson_goodness_of_fit(data, lambda_mle):
                return "Poisson"
            
            # Geometric test
            p_mle = 1 / (np.mean(data) + 1)
            if self.geometric_goodness_of_fit(data, p_mle):
                return "Geometric"
        
        # Continuous distributions
        # Normal test
        _, p_value = stats.normaltest(data)
        if p_value > 0.05:
            return "Normal"
        
        # Uniform test
        if self.uniform_goodness_of_fit(data):
            return "Uniform"
        
        # Exponential test
        if np.all(data > 0) and self.exponential_goodness_of_fit(data):
            return "Exponential"
        
        # Lognormal test
        if np.all(data > 0):
            _, p_value = stats.normaltest(np.log(data))
            if p_value > 0.05:
                return "Lognormal"
        
        return "Unknown"

    def poisson_goodness_of_fit(self, data: np.ndarray, lambda_mle: float) -> bool:
        observed = np.bincount(data.astype(int))
        expected = stats.poisson.pmf(np.arange(len(observed)), lambda_mle) * len(data)
        chi_square, p_value = stats.chisquare(observed, expected)
        return p_value > 0.05

    def geometric_goodness_of_fit(self, data: np.ndarray, p_mle: float) -> bool:
        observed = np.bincount(data.astype(int))
        expected = stats.geom.pmf(np.arange(1, len(observed) + 1), p_mle) * len(data)
        chi_square, p_value = stats.chisquare(observed, expected)
        return p_value > 0.05

    def uniform_goodness_of_fit(self, data: np.ndarray) -> bool:
        _, p_value = stats.kstest(data, 'uniform', args=(np.min(data), np.max(data) - np.min(data)))
        return p_value > 0.05

    def exponential_goodness_of_fit(self, data: np.ndarray) -> bool:
        _, p_value = stats.kstest(data, 'expon', args=(0, np.mean(data)))
        return p_value > 0.05

    def preprocess_distribution(self, data: np.ndarray, dist_type: str) -> np.ndarray:
        if dist_type == "Poisson":
            return np.sqrt(data)
        elif dist_type == "Geometric":
            return np.log1p(data)
        elif dist_type == "Normal":
            return self.standardize(data)
        elif dist_type == "Uniform":
            return self.normalize(data)
        elif dist_type == "Exponential" or dist_type == "Lognormal":
            pt = PowerTransformer(method='box-cox', standardize=True)
            return pt.fit_transform(data.reshape(-1, 1)).flatten()
        else:  # Unknown distribution
            return self.standardize(data)
        

    def analyze_column(self, data: np.ndarray, is_timeseries: bool = False) -> Dict:
        print(f"\nAnalyzing column (is_timeseries={is_timeseries}):")
        results = {}
        data = self.impute_missing_values(data)
        
        print("  Calculating basic statistics...")
        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['skew'] = stats.skew(data)
        results['kurtosis'] = stats.kurtosis(data)
        results['outliers'] = self.detect_outliers(data)
        results['unique_count'] = len(np.unique(data))
        results['missing_ratio'] = np.sum(np.isnan(data)) / len(data)

        if is_timeseries:
            print("  Performing time series analysis...")
            results['adf_test'] = self.augmented_dickey_fuller(data)
            results['acf'] = self.autocorrelation(data, 10)
            results['pacf'] = self.partial_autocorrelation(data, 10)
            results['heteroscedasticity'] = self.breusch_pagan_test(data)
            results['lyapunov'] = self.largest_lyapunov_exponent(data, self.config['embedding_dim'], self.config['tau'], self.config['lyap_k'])
            results['correlation_dim'] = self.correlation_dimension(data, self.config['embedding_dim'], self.config['tau'], 0.1)
            
            q_range = np.linspace(-self.config['q_order'], self.config['q_order'], 20)
            box_sizes = np.logspace(1, np.log10(len(data)//4), 20, dtype=int)
            results['mf_spectrum'] = self.multifractal_spectrum(data, q_range, box_sizes)
            
            results['hurst'] = self.hurst_exponent(data, self.config['hurst_lags'])
            results['sample_entropy'] = self.sample_entropy(data, m=2, r=0.2*np.std(data))
        else:
            print("  Detecting distribution...")
            results['distribution'] = self.detect_distribution(data)
            print(f"  Detected distribution: {results['distribution']}")
            results['normality'] = stats.normaltest(data)[1]
            results['correlation_matrix'] = np.corrcoef(data.reshape(-1, 1).T, data.reshape(-1, 1))[0, 1]
        
        print("  Analysis complete.")
        return results

    def get_insights(self, results: Dict, is_timeseries: bool = False) -> List[str]:
        insights = []
        
        if is_timeseries:
            if results['adf_test'] > 0.05:
                insights.append("Non-stationary")
            else:
                insights.append("Stationary")
            
            if np.max(np.abs(results['acf'][1:])) > 1.96 / np.sqrt(len(results['acf'])):
                insights.append("Significant autocorrelation")
            
            if results['heteroscedasticity'] < 0.05:
                insights.append("Heteroscedastic")
            else:
                insights.append("Homoscedastic")
            
            if results['lyapunov'] > 0:
                insights.append("Chaotic")
            else:
                insights.append("Non-chaotic")
            
            if results['hurst'] > 0.5:
                insights.append("Long-range dependence")
            elif results['hurst'] < 0.5:
                insights.append("Anti-persistent")
            else:
                insights.append("Random walk")
        else:
            if results['normality'] < 0.05:
                insights.append("Non-normal distribution")
            else:
                insights.append("Normal distribution")
            
            if abs(results['correlation_matrix']) > 0.7:
                insights.append("Strong linear correlation")
            elif abs(results['correlation_matrix']) > 0.3:
                insights.append("Moderate linear correlation")
            else:
                insights.append("Weak linear correlation")
        
        if abs(results['skew']) > 0.5:
            insights.append("Skewed")
        
        if results['kurtosis'] > 3:
            insights.append("Heavy-tailed")
        elif results['kurtosis'] < 3:
            insights.append("Light-tailed")
        
        if np.sum(results['outliers']) > 0:
            insights.append("Contains outliers")
        
        if results['unique_count'] / len(results['outliers']) < 0.1:
            insights.append("Low cardinality")
        
        if results['missing_ratio'] > 0:
            insights.append(f"Contains missing values ({results['missing_ratio']:.2%})")
        
        return insights

    def preprocess_column(self, data: np.ndarray, insights: List[str], is_timeseries: bool = False) -> np.ndarray:
        data = self.impute_missing_values(data)
        
        if "Contains outliers" in insights:
            outliers = self.detect_outliers(data)
            data = data[~outliers]
        
        
        if is_timeseries:
            if "Non-stationary" in insights:
                data = np.diff(data)
                data = np.insert(data, 0, data[0])
            
            if "Significant autocorrelation" in insights:
                data = self.remove_trend(data)
                if len(data) >= 12:
                    _, _, data = seasonal_decompose(data, model='additive', period=12).resid
            
            if "Heteroscedastic" in insights:
                data = np.log1p(data - np.min(data) + 1e-8)
        else:
            dist_type = next((insight for insight in insights if "Distribution: " in insight), None)
            if dist_type:
                dist_type = dist_type.split(": ")[1]
                data = self.preprocess_distribution(data, dist_type)
            else:
                data = self.standardize(data)
        
        return data

    @staticmethod
    def is_numeric(data: np.ndarray) -> bool:
        return np.issubdtype(data.dtype, np.number)

    @staticmethod
    def is_categorical(data: np.ndarray) -> bool:
        return data.dtype == 'object' or data.dtype.name == 'category'

    @staticmethod
    def is_datetime(data: np.ndarray) -> bool:
        return np.issubdtype(data.dtype, np.datetime64)

    @staticmethod
    def is_id_column(data: np.ndarray) -> bool:
        if not EnhancedMathAutopreprocessor.is_numeric(data):
            return False
        unique_values = np.unique(data)
        return len(unique_values) == len(data) and np.allclose(np.diff(np.sort(unique_values)), 1)
    @staticmethod
    def encode_categorical(data: np.ndarray) -> np.ndarray:
        unique_values = np.unique(data)
        encoding_dict = {val: i for i, val in enumerate(unique_values)}
        return np.array([encoding_dict[val] for val in data])

    @staticmethod
    def encode_datetime(data: np.ndarray) -> np.ndarray:
        return data.astype(np.int64) // 10**9  # Convert to Unix timestamp
    def run(self, data: np.ndarray, time_columns: List[int] = None, target: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        preprocessed_data = np.zeros_like(data, dtype=float)
        all_insights = {}
        
        time_columns = time_columns or []
        
        for i in range(data.shape[1]):
            column_data = data[:, i]
            is_timeseries = i in time_columns
            
            if self.is_id_column(column_data):
                preprocessed_data[:, i] = column_data
                all_insights[f'column_{i}'] = ["ID column - not preprocessed"]
                continue
            
            if not self.is_numeric(column_data):
                if self.is_categorical(column_data):
                    preprocessed_data[:, i] = self.encode_categorical(column_data)
                    all_insights[f'column_{i}'] = ["Categorical data - encoded"]
                elif self.is_datetime(column_data):
                    preprocessed_data[:, i] = self.encode_datetime(column_data)
                    all_insights[f'column_{i}'] = ["Datetime data - encoded"]
                else:
                    preprocessed_data[:, i] = column_data
                    all_insights[f'column_{i}'] = ["Non-numeric data - not preprocessed"]
                continue
            
            results = self.analyze_column(column_data, is_timeseries)
            insights = self.get_insights(results, is_timeseries)
            preprocessed_data[:, i] = self.preprocess_column(column_data, insights, is_timeseries)
            all_insights[f'column_{i}'] = insights
        
        preprocessed_data = self.standardize(preprocessed_data)
        
        if target is not None:
            preprocessed_data, removed_columns = self.handle_multicollinearity(preprocessed_data, target)
            all_insights['removed_columns'] = removed_columns
            
            mi_scores = mutual_info_regression(preprocessed_data, target)
            all_insights['mutual_information'] = mi_scores.tolist()
        
        pca = PCA(n_components=self.config['pca_threshold'], svd_solver='full')
        pca_data = pca.fit_transform(preprocessed_data)
        all_insights['pca_explained_variance'] = np.sum(pca.explained_variance_ratio_)
        
        poly = PolynomialFeatures(degree=self.config['polynomial_degree'])
        poly_features = poly.fit_transform(preprocessed_data)
        all_insights['polynomial_features_shape'] = poly_features.shape
        
        kmeans = KMeans(n_clusters=min(5, preprocessed_data.shape[0]//10))
        clusters = kmeans.fit_predict(preprocessed_data)
        all_insights['cluster_labels'] = clusters.tolist()
        
        return preprocessed_data, all_insights

    def process_dataframe(self, df: Union[pd.DataFrame, np.ndarray], time_columns: List[str] = None, target_column: str = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Dict]:
        if isinstance(df, pd.DataFrame):
            data = df.values
            column_names = df.columns
            time_column_indices = [df.columns.get_loc(col) for col in (time_columns or [])]
            
            if target_column:
                target = df[target_column].values
                data = df.drop(columns=[target_column]).values
            else:
                target = None
        else:  # Assume it's a numpy array
            data = df
            column_names = [f'column_{i}' for i in range(data.shape[1])]
            time_column_indices = time_columns if time_columns else []
            
            if target_column is not None:
                target = data[:, target_column]
                data = np.delete(data, target_column, axis=1)
            else:
                target = None

        preprocessed_data, all_insights = self.run(data, time_column_indices, target)
        
        if isinstance(df, pd.DataFrame):
            preprocessed_df = pd.DataFrame(preprocessed_data, columns=column_names)
            if target_column:
                preprocessed_df[target_column] = target
            return preprocessed_df, all_insights
        else:
            if target is not None:
                preprocessed_data = np.column_stack((preprocessed_data, target))
            return preprocessed_data, all_insights

    def suggest_models(self, insights: Dict, is_timeseries: bool, task_type: str) -> List[str]:
        suggested_models = []
        
        if is_timeseries:
            suggested_models.extend([
                "ARIMA", "SARIMA", "Prophet", "LSTM"
            ])
            if "Chaotic" in insights:
                suggested_models.extend(["Echo State Networks", "Reservoir Computing"])
            if "Long-range dependence" in insights:
                suggested_models.append("ARFIMA")
        else:
            if task_type == "regression":
                suggested_models.extend([
                    "Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost"
                ])
                if "Non-normal distribution" in insights or "Skewed" in insights:
                    suggested_models.append("Quantile Regression")
                if "Strong linear correlation" not in insights:
                    suggested_models.extend(["Support Vector Regression", "Neural Networks"])
            elif task_type == "classification":
                suggested_models.extend([
                    "Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier", "XGBoost"
                ])
                if "Non-normal distribution" in insights or "Skewed" in insights:
                    suggested_models.append("Decision Trees")
                if "Strong linear correlation" not in insights:
                    suggested_models.extend(["Support Vector Machines", "Neural Networks"])
        
        if "Low cardinality" in insights:
            suggested_models.append("CatBoost")
        
        return suggested_models
    def generate_insights_report(self, all_insights: Dict, is_timeseries: bool, task_type: str) -> str:
        report = "# Insight Report\n\n"
        
        report += "## Column-wise Insights\n\n"
        for column, insights in all_insights.items():
            if column not in ["removed_columns", "mutual_information", "pca_explained_variance", "polynomial_features_shape", "cluster_labels"]:
                report += f"### {column}\n"
                for insight in insights:
                    report += f"- {insight}\n"
                report += "\n"
        
        report += "## Overall Insights\n\n"
        if "removed_columns" in all_insights:
            report += f"- Removed columns due to multicollinearity: {all_insights['removed_columns']}\n"
        if "mutual_information" in all_insights:
            report += f"- Mutual Information scores: {all_insights['mutual_information']}\n"
        if "pca_explained_variance" in all_insights:
            report += f"- PCA Explained Variance: {all_insights['pca_explained_variance']:.2f}\n"
        if "polynomial_features_shape" in all_insights:
            report += f"- Polynomial Features Shape: {all_insights['polynomial_features_shape']}\n"
        if "cluster_labels" in all_insights:
            report += f"- Number of Clusters: {len(set(all_insights['cluster_labels']))}\n"
        
        report += "\n## Preprocessing Steps Applied\n\n"
        report += "1. Imputation of missing values\n"
        report += "2. Outlier detection and removal\n"
        if is_timeseries:
            report += "3. Stationarity transformation for time series data\n"
            report += "4. Trend and seasonality removal for time series data\n"
        report += "5. Normalization of non-normal distributions\n"
        report += "6. Box-Cox transformation for skewed and heavy-tailed data\n"
        report += "7. Standardization of all numeric columns\n"
        report += "8. Encoding of categorical variables\n"
        report += "9. Multicollinearity handling\n"
        report += "10. Dimensionality reduction using PCA\n"
        report += "11. Generation of polynomial features\n"
        report += "12. Cluster analysis\n"
        
        report += "\n## Suggested Models\n\n"
        suggested_models = self.suggest_models(all_insights, is_timeseries, task_type)
        for model in suggested_models:
            report += f"- {model}\n"
        
        report += "\n## Recommendations\n\n"
        report += "1. Consider feature importance analysis to identify the most relevant features.\n"
        report += "2. Experiment with different combinations of preprocessing steps to optimize model performance.\n"
        if is_timeseries:
            report += "3. For time series data, pay attention to lag selection and consider using cross-validation techniques specific to time series.\n"
        else:
            report += "3. For non-time series data, consider using cross-validation techniques to ensure model generalization.\n"
        report += "4. If dealing with imbalanced classes in classification tasks, consider using techniques like SMOTE or class weighting.\n"
        report += "5. Regularly monitor and update your models, especially for time series data where patterns may change over time.\n"
        
        return report

    def full_pipeline(self, df: Union[pd.DataFrame, np.ndarray], time_columns: List[str] = None, target_column: str = None, task_type: str = "regression") -> Tuple[Union[pd.DataFrame, np.ndarray], Dict, str, List[str]]:
        preprocessed_df, all_insights = self.process_dataframe(df, time_columns, target_column)
        is_timeseries = bool(time_columns)
        insights_report = self.generate_insights_report(all_insights, is_timeseries, task_type)
        
        # Extract removed columns from all_insights
        removed_columns = all_insights.get('removed_columns', [])
        
        return preprocessed_df, all_insights, insights_report, removed_columns


if __name__ == "__main__":
    # Read the sample data from CSV
    df = pd.read_csv('possum.csv')
    
    # Check if 'Target' column exists
    target = None
    if 'Target' in df.columns:
        target = df.pop('Target').values
    
    # Initialize the preprocessor
    preprocessor = EnhancedMathAutopreprocessor()
    
    print("Starting full pipeline...")
    # Run full pipeline
    preprocessed_data, raw_insights, insights_report, removed_columns = preprocessor.full_pipeline(df, target_column='Target', task_type="regression")
    
    print("\nFull pipeline complete.")
    print("Original data shape:", df.shape)
    print("Preprocessed data shape:", preprocessed_data.shape)
    print("Removed columns:", removed_columns)
    
    # Save the preprocessed data
    pd.DataFrame(preprocessed_data, columns=[f'feature_{i}' for i in range(preprocessed_data.shape[1])]).to_csv('preprocessed_data.csv', index=False)
    print("Preprocessed data has been saved to 'preprocessed_data.csv'")
    
    # Display insights report
    print("\nInsights Report:")
    print("================")
    print(insights_report)
    
    # Visualize original vs preprocessed data
    n_features = min(5, df.shape[1])  # Limit to 5 features for brevity
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
    fig.suptitle("Original vs Preprocessed Data (First 5 Features)")
    
    for i, column in enumerate(df.columns[:n_features]):
        if df[column].dtype != 'object' and not pd.api.types.is_datetime64_any_dtype(df[column]):
            sns.histplot(df[column], ax=axes[i, 0], kde=True)
            axes[i, 0].set_title(f"Original: {column}")
            
            sns.histplot(preprocessed_data[:, i], ax=axes[i, 1], kde=True)
            axes[i, 1].set_title(f"Preprocessed: feature_{i}")
        else:
            axes[i, 0].text(0.5, 0.5, f"Non-numeric column: {column}", ha='center', va='center')
            axes[i, 1].text(0.5, 0.5, f"Non-numeric column: feature_{i}", ha='center', va='center')
    
    plt.tight_layout()
    plt.show()