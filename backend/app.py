import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import pandas as pd
from io import StringIO, BytesIO
import tempfile
import os
import logging
import requests
from logging.handlers import RotatingFileHandler
from helpers import *
from utils import perform_eda, AdvancedPreprocessor
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
ENCODINGS = ['utf-8', 'latin-1', 'ascii', 'iso-8859-1', 'cp1252']

# Set up logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

@app.route('/preprocess', methods=['GET'])
def preprocess_data():
    """
    API endpoint for preprocessing CSV data.

    This endpoint supports the GET method, which accepts a CSV file URL as a query parameter.

    The function performs the following steps:
    1. Receives the CSV file URL via a query parameter.
    2. Downloads the CSV data and loads it into a pandas DataFrame.
    3. Applies advanced preprocessing techniques using the AdvancedPreprocessor.
    4. Returns the preprocessed data as a downloadable CSV file.

    Returns:
        - A CSV file attachment containing the preprocessed data.

    In case of an error:
        - Returns a JSON response with an error message and the appropriate HTTP status code.

    Raises:
        - 400 Bad Request: If the URL is missing or the file cannot be downloaded.
        - 500 Internal Server Error: If an error occurs during preprocessing.
    """
    try:
        app.logger.info("Received preprocessing request")
        url = request.args.get('url')
        if not url:
            raise ValueError("No URL provided")
        
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError("Failed to download the file from the URL")
        
        content = response.content
        # Use BytesIO to wrap the binary content
        csv_buffer = BytesIO(content)
        df = pd.read_csv(csv_buffer)
        
        app.logger.info("Preprocessing data")
        preprocessor = AdvancedPreprocessor()
        preprocessed_df = preprocessor.preprocess(df)

        # Create a CSV from the preprocessed DataFrame
        output_buffer = BytesIO()
        preprocessed_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)  # Move the pointer to the beginning of the BytesIO object

        app.logger.info("Preprocessing completed successfully")
        return send_file(
            output_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='preprocessed_data.csv'
        )
    except ValueError as ve:
        app.logger.error(f"Preprocessing error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error during preprocessing: {str(e)}")
        return jsonify({"error": f"Error during preprocessing: {str(e)}"}), 500

@app.route('/visualize', methods=['POST', 'GET'])
def visualize_data():
    """
    API endpoint for visualizing CSV data.
    
    This endpoint supports both POST and GET methods:
    - POST: Accepts a file upload containing CSV data.
    - GET: Accepts CSV data as a query parameter.
    
    The function performs the following steps:
    1. Receives the data (either via file upload or query parameter)
    2. Parses the CSV data into a pandas DataFrame
    3. Performs Exploratory Data Analysis (EDA) using the perform_eda function
    4. Generates an HTML report with the visualizations
    5. Returns the HTML content in a JSON response
    
    If the generated HTML content is too large (>10MB), it returns a message
    indicating that the content is too large to be returned directly.
    
    Returns:
        JSON response with the following structure:
        {
            "message": "Visualization completed successfully",
            "html_content": "<HTML content of the visualization>"
        }
        
        Or, if the content is too large:
        {
            "message": "Visualization completed successfully, but content is too large to return directly",
            "error": "Content exceeds size limit"
        }
        
        In case of an error:
        {
            "error": "<Error message describing the issue>"
        }
    
    Raises:
        400 Bad Request: If there's an issue with the input data or file.
        500 Internal Server Error: If there's an error during visualization.
    """
    try:
        app.logger.info("Received visualization request")
        if request.method == 'POST':
            df = handle_file_upload(request)
        elif request.method == 'GET':
            df = handle_get_request(request)
        
        app.logger.info("Performing EDA and generating visualizations")
        eda_results = perform_eda(df)
        html_content = generate_html_report(df, eda_results)
        
        if not html_content:
            app.logger.error("HTML content generation failed")
            return jsonify({"error": "HTML content generation failed"}), 500

        if len(html_content) > 30_000_000:  # 30 MB limit
            app.logger.info("Visualization content too large to return directly")
            return jsonify({
                "message": "Visualization completed successfully, but content is too large to return directly",
                "error": "Content exceeds size limit"
            }), 413  # 413 Payload Too Large

        app.logger.info("Visualization completed successfully")
        return jsonify({
            "message": "Visualization completed successfully",
            "html_content": html_content
        })
    except ValueError as ve:
        app.logger.error(f"Visualization error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error during visualization: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error during visualization: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)