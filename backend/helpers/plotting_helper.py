import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
