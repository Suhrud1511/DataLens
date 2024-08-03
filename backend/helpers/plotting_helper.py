import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#7FDBFF", "#F012BE", "#01FF70", "#FFD700", "#B10DC9"]

def plot_to_image(fig, file_format='png'):
    """
    Convert a Matplotlib figure to an image encoded in base64.

    This function is useful for embedding plots in HTML or other formats that support base64-encoded images.

    Parameters:
    fig (matplotlib.figure.Figure): The Matplotlib figure to convert.
    file_format (str): The format of the output image (default is 'png').

    Returns:
    str: A string containing the base64-encoded image.
    """
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



def correlation_analysis(df, numerical_columns):
    """
    Perform correlation analysis on numerical columns of a DataFrame.

    This function calculates the correlation matrix for the given numerical columns.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to analyze.

    Returns:
    pandas.DataFrame: The correlation matrix of the numerical columns.
    """
    print("Performing correlation analysis.")
    try:
        correlation_matrix = df[numerical_columns].corr()
        return correlation_matrix
    except Exception as e:
        print(f"Error performing correlation analysis: {e}")
        return None

def distribution_analysis(df, numerical_columns):
    """
    Generate distribution plots for numerical columns in a DataFrame.

    This function creates histogram plots with KDE for each numerical column.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each distribution plot.
    """
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
    """
    Generate boxplots for numerical columns in a DataFrame.

    This function creates a boxplot for each numerical column to visualize the distribution and identify outliers.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each boxplot.
    """
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
    """
    Generate a heatmap for the correlation matrix.

    This function creates a heatmap to visualize the correlations between numerical columns.

    Parameters:
    correlation_matrix (pandas.DataFrame): The correlation matrix to visualize.

    Returns:
    str: A base64-encoded image of the heatmap.
    """
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
    """
    Generate a pairplot for numerical columns in a DataFrame.

    This function creates a pairplot to visualize pairwise relationships and distributions of numerical columns.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    str: A base64-encoded image of the pairplot.
    """
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
    """
    Generate a heatmap to visualize missing values in the DataFrame.

    This function creates a heatmap where missing values are highlighted.

    Parameters:
    df (pandas.DataFrame): The input data.

    Returns:
    str: A base64-encoded image of the missing values heatmap.
    """
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
    """
    Perform outlier analysis using boxplots for numerical columns.

    This function creates boxplots for each numerical column to visualize and identify outliers.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each outlier plot.
    """
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
    """
    Generate KDE plots for numerical columns in a DataFrame.

    This function creates Kernel Density Estimate (KDE) plots for each numerical column.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each KDE plot.
    """
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
    """
    Generate violin plots for numerical columns in a DataFrame.

    This function creates violin plots to visualize the distribution of numerical columns.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each violin plot.
    """
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
    """
    Generate joint plots for numerical columns in a DataFrame.

    This function creates joint plots for numerical columns, showing scatter plots and marginal distributions.

    Parameters:
    df (pandas.DataFrame): The input data.
    numerical_columns (list of str): The list of numerical columns to plot.

    Returns:
    list of str: A list of base64-encoded images for each joint plot.
    """
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
