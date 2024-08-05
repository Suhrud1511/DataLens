
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import pandas as pd
from io import StringIO
import tempfile
import os
import logging
from logging.handlers import RotatingFileHandler
from helpers import *
from utils import perform_eda, AdvancedPreprocessor
import traceback

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
ENCODINGS = ['utf-8', 'latin-1', 'ascii', 'iso-8859-1', 'cp1252']

# Set up logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The name of the uploaded file.
    
    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_csv_with_encodings(file):
    """
    Attempt to read a CSV file with multiple encodings.
    
    This function tries to read the CSV file using various encodings to handle
    different file formats and character sets.
    
    Args:
        file: File-like object containing CSV data.
    
    Returns:
        pandas.DataFrame: The parsed CSV data.
    
    Raises:
        ValueError: If the file can't be decoded with any supported encoding.
    """
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(file, encoding=encoding)
            app.logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            app.logger.warning(f"Failed to decode with encoding: {encoding}")
            continue
    raise ValueError("Unable to decode the CSV file with any of the supported encodings.")

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess_data():
    """
    API endpoint for preprocessing CSV data.
    
    This endpoint supports both POST and GET methods:
    - POST: Accepts a file upload containing CSV data.
    - GET: Accepts CSV data as a query parameter.
    
    The function performs the following steps:
    1. Receives the data (either via file upload or query parameter)
    2. Parses the CSV data into a pandas DataFrame
    3. Applies advanced preprocessing techniques using the AdvancedPreprocessor
    4. Returns the preprocessed data as a CSV string in a JSON response
    
    Returns:
        JSON response with the following structure:
        {
            "message": "Preprocessing completed successfully",
            "preprocessed_data": "<CSV string of preprocessed data>"
        }
        
        In case of an error:
        {
            "error": "<Error message describing the issue>"
        }
    
    Raises:
        400 Bad Request: If there's an issue with the input data or file.
        500 Internal Server Error: If there's an error during preprocessing.
    """
    try:
        app.logger.info("Received preprocessing request")
        if request.method == 'POST':
            df = handle_file_upload(request)
        elif request.method == 'GET':
            df = handle_get_request(request)
        
        app.logger.info("Preprocessing data")
        preprocessor = AdvancedPreprocessor()
        preprocessed_df = preprocessor.preprocess(df)

        csv_buffer = StringIO()
        preprocessed_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        app.logger.info("Preprocessing completed successfully")
        return jsonify({
            "message": "Preprocessing completed successfully",
            "preprocessed_data": csv_string
        })
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

def handle_file_upload(request):
    """
    Handle file upload for POST requests.
    
    This function processes the file upload, checks for common errors,
    and reads the CSV data into a pandas DataFrame.
    
    Args:
        request: Flask request object containing the file upload.
    
    Returns:
        pandas.DataFrame: Parsed CSV data.
    
    Raises:
        ValueError: If file upload is missing, empty, or has an invalid format.
    """
    if 'file' not in request.files:
        raise ValueError("No file part in the request")
    
    file = request.files['file']
    
    if file.filename == '':
        raise ValueError("No selected file")
    
    if not allowed_file(file.filename):
        raise ValueError("Invalid file format. Please upload a CSV file.")
    
    app.logger.info(f"Processing uploaded file: {file.filename}")
    return read_csv_with_encodings(file)

def handle_get_request(request):
    """
    Handle GET requests with CSV data as a parameter.
    
    This function extracts CSV data from the request parameters and
    parses it into a pandas DataFrame.
    
    Args:
        request: Flask request object containing the CSV data as a parameter.
    
    Returns:
        pandas.DataFrame: Parsed CSV data.
    
    Raises:
        ValueError: If CSV data is missing or cannot be parsed.
    """
    csv_data = request.args.get('data')
    if not csv_data:
        raise ValueError("No CSV data provided in the request parameters")
    
    app.logger.info("Processing CSV data from GET request")
    return pd.read_csv(StringIO(csv_data))

@app.route('/')
def index():
    """
    Render the main page with file upload forms.
    
    This function generates an HTML page with forms for data preprocessing
    and visualization, allowing users to upload CSV files for processing.
    
    Returns:
        str: HTML content of the main page.
    """
    app.logger.info("Rendering index page")
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Preprocessing and Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            h1 { color: #333; }
            form { margin-bottom: 20px; }
            input[type="file"] { margin-bottom: 10px; }
            input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
            input[type="submit"]:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <h1>Data Preprocessing and Visualization</h1>
        <h2>Preprocess Data</h2>
        <form action="/preprocess" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv">
            <input type="submit" value="Preprocess">
        </form>
        <h2>Visualize Data</h2>
        <form action="/visualize" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv">
            <input type="submit" value="Visualize">
        </form>
    </body>
    </html>
    '''
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)