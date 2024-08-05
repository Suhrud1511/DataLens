# helpers/__init__.py

from .html_generator import generate_html_report
# from .preprocess_helper import preprocess
from .plotting_helper import ( 
    plot_to_image,
    correlation_analysis,
    distribution_analysis,
    boxplot_analysis,
    heatmap_analysis,
    pairplot_analysis,
    missing_values_heatmap,
    outlier_analysis,
    kde_analysis,
    violin_plot_analysis,
    joint_plot_analysis)

from .insight_helper import (    
    data_quality_score,
    generate_automated_insights)

from .text_analysis_helper import text_data_analysis
from .geospatial_analysis_helper import geospatial_analysis

from .statistical_helper import (
    
    identify_columns,
    analyze_categorical,
    analyze_numerical,
    determine_scaling_technique,
    feature_importance_analysis,
    time_series_analysis,
    dimensionality_reduction,
    anomaly_detection,
    statistical_tests
    
)

# Define an `__all__` list to control what gets imported with `from helpers import *`
__all__ = [
    'plot_to_image',
    'identify_columns',
    'analyze_categorical',
    'analyze_numerical',
    'determine_scaling_technique',
    'correlation_analysis',
    'distribution_analysis',
    'boxplot_analysis',
    'heatmap_analysis',
    'pairplot_analysis',
    'missing_values_heatmap',
    'outlier_analysis',
    'kde_analysis',
    'violin_plot_analysis',
    'joint_plot_analysis',
    'feature_importance_analysis',
    'time_series_analysis',
    'dimensionality_reduction',
    'anomaly_detection',
    'statistical_tests',
    'text_data_analysis',
    'geospatial_analysis',
    'data_quality_score',
    'generate_automated_insights',
    # 'preprocess',
    'generate_html_report'

]

