from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
from ydata_profiling import ProfileReport
from datetime import datetime
from flask_cors import CORS
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/reports'
HEATMAP_FOLDER = 'static/heatmaps'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

LAST_UPLOADED_DF = None  


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global LAST_UPLOADED_DF
    file = request.files['dataset']
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)
    LAST_UPLOADED_DF = df.copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eda_report_{timestamp}.html"
    report_path = os.path.join(UPLOAD_FOLDER, filename)

    profile = ProfileReport(df, title="Exploratory Data Analysis Report", explorative=True)
    profile.to_file(report_path)

    return {"filename": filename}


@app.route('/preprocess', methods=['POST'])
def preprocess():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return "No dataset uploaded yet", 400

    df = LAST_UPLOADED_DF.copy()

    import json
    options = json.loads(request.form.get('options'))

    if options.get("missing") == "mean":
        imputer = MeanMedianImputer(imputation_method="mean")
        df = imputer.fit_transform(df)
    elif options.get("missing") == "median":
        imputer = MeanMedianImputer(imputation_method="median")
        df = imputer.fit_transform(df)

   
    if options.get("encode") == "label":
        encode_columns = options.get("encode_columns", [])
        for col in encode_columns:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    
    scaling = options.get("scaling")
    scale_columns = options.get("scale_columns", [])
    if scaling in ["standard", "minmax"] and scale_columns:
        scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()
        for col in scale_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = scaler.fit_transform(df[[col]])

    
    # numeric_cols = df.select_dtypes(include='number').columns
    # for col in numeric_cols:
    #     Q1 = df[col].quantile(0.25)
    #     Q3 = df[col].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    
    LAST_UPLOADED_DF = df.copy()

    return jsonify(df.head(10).replace({np.nan: None}).to_dict(orient='records'))


@app.route('/correlation')
def correlation():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return "No dataset uploaded yet", 400

    df = LAST_UPLOADED_DF.copy()
    df = df.select_dtypes(include='number')

    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')

    if df.empty or df.shape[1] < 2:
        return jsonify({
            "correlation_table": {},
            "heatmap_path": None,
            "error": "Not enough numeric data to compute correlation."
        })

    corr_matrix = df.corr()

    corr_matrix_clean = corr_matrix.replace({np.nan: 0.0})

   
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_clean, annot=True, cmap='coolwarm', fmt='.2f')
    heatmap_path = os.path.join(HEATMAP_FOLDER, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return jsonify({
        "correlation_table": corr_matrix.replace({np.nan: None}).to_dict(),
        "heatmap_path": f"/static/heatmaps/correlation_heatmap.png"
    })


@app.route('/columns')
def get_columns():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is not None:
        return jsonify({"columns": list(LAST_UPLOADED_DF.columns)})
    return jsonify({"columns": []})


@app.route('/download', methods=['POST'])
def download():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return "No dataset available", 400

    import json
    selected = request.get_json().get("selected_columns", [])
    filtered_df = LAST_UPLOADED_DF[selected]
    output_path = 'static/downloads/filtered_dataset.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
