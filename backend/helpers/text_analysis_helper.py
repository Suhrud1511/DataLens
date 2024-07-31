import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from .plotting_helper import plot_to_image
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
