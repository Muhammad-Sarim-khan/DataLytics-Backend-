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
FINAL_PROCESSED_DF = None  

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
    global LAST_UPLOADED_DF, FINAL_PROCESSED_DF
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

    outlier_columns = options.get("outlier_columns", [])
    for col in outlier_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    FINAL_PROCESSED_DF = df.copy()  
    LAST_UPLOADED_DF = df.copy()   
    return jsonify(df.head(10).replace({np.nan: None}).to_dict(orient='records'))

@app.route('/correlation')
def correlation():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return "No dataset uploaded yet", 400

    df = LAST_UPLOADED_DF.copy()

    numeric_df = df.select_dtypes(include='number').copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna(axis=1, how='all')

    if numeric_df.empty or numeric_df.shape[1] < 2:
        return jsonify({
            "correlation_table": {},
            "heatmap_path": None,
            "numeric_columns": list(numeric_df.columns),
            "selectable_columns": list(df.columns),  
            "error": "Not enough numeric data to compute correlation."
        })

    corr_matrix = numeric_df.corr()
    corr_matrix_clean = corr_matrix.replace({np.nan: 0.0})

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_clean, annot=True, cmap='coolwarm', fmt='.2f')
    heatmap_path = os.path.join(HEATMAP_FOLDER, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return jsonify({
        "correlation_table": corr_matrix.replace({np.nan: None}).to_dict(),
        "heatmap_path": f"/static/heatmaps/correlation_heatmap.png",
        "numeric_columns": list(numeric_df.columns),
        "selectable_columns": list(df.columns)  
    })



@app.route('/column_metadata')
def column_metadata():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return jsonify([])

    df = LAST_UPLOADED_DF.copy()
    metadata = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())

        outliers = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = int(((df[col] < lower) | (df[col] > upper)).sum())

        metadata.append({
            "column": col,
            "nulls": nulls,
            "outliers": outliers,
            "dtype": dtype
        })

    return jsonify(metadata)

@app.route('/columns')
def get_columns():
    global FINAL_PROCESSED_DF
    if FINAL_PROCESSED_DF is None:
        return jsonify({})

    df = FINAL_PROCESSED_DF.copy()
    total_rows = len(df)  

    col_metadata = {}

    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        outliers = 0

        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = int(((df[col] < lower) | (df[col] > upper)).sum())

        col_metadata[col] = {
            "nulls": nulls,
            "outliers": outliers,
            "dtype": dtype,
            "total_rows": total_rows  
        }

    return jsonify(col_metadata)



@app.route('/remove_column', methods=['POST'])
def remove_column():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return jsonify({"error": "No dataset available"}), 400

    data = request.get_json()
    column_to_remove = data.get("column")

    if column_to_remove not in LAST_UPLOADED_DF.columns:
        return jsonify({"error": f"Column '{column_to_remove}' not found"}), 404

    LAST_UPLOADED_DF.drop(columns=[column_to_remove], inplace=True)
    return jsonify({
        "message": f"Column '{column_to_remove}' removed",
        "remaining_columns": LAST_UPLOADED_DF.columns.tolist()
    })

@app.route('/download', methods=['POST'])
def download():
    global FINAL_PROCESSED_DF
    if FINAL_PROCESSED_DF is None:
        return "No dataset available", 400

    import json
    selected = request.get_json().get("selected_columns", [])
    filtered_df = FINAL_PROCESSED_DF[selected]
    output_path = 'static/downloads/filtered_dataset.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    return send_file(output_path, as_attachment=True)


@app.route('/correlation_pair', methods=['POST'])
def correlation_pair():
    global LAST_UPLOADED_DF
    if LAST_UPLOADED_DF is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    data = request.get_json()
    target = data.get("target")
    feature = data.get("feature")

    if target not in LAST_UPLOADED_DF.columns or feature not in LAST_UPLOADED_DF.columns:
        return jsonify({"error": "Invalid column names"}), 400

    df = LAST_UPLOADED_DF.copy()
    df = df[[target, feature]].dropna()

    df[target] = pd.to_numeric(df[target], errors='coerce')
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    df = df.dropna()

    if df.empty:
        return jsonify({"correlation": None, "error": "Insufficient numeric data"}), 200

    correlation = df[target].corr(df[feature])
    return jsonify({
        "target": target,
        "feature": feature,
        "correlation": round(correlation, 4)
    })

@app.route('/final_dataset')
def final_dataset():
    global FINAL_PROCESSED_DF
    if FINAL_PROCESSED_DF is None:
        return jsonify({"error": "No dataset available"}), 400

    features = request.args.getlist('features')
    if not features:
        features_str = request.args.get('features')
        if features_str:
            features = features_str.split(',')
    df = FINAL_PROCESSED_DF
    if features:
        features = [f for f in features if f in df.columns]
        df = df[features]
    return jsonify(df.replace({np.nan: None}).to_dict(orient='records'))



@app.route('/selected_columns')
def selected_columns():
    global FINAL_PROCESSED_DF
    if FINAL_PROCESSED_DF is None:
        return jsonify([])

    return jsonify(FINAL_PROCESSED_DF.columns.tolist())

if __name__ == '__main__':
    app.run(debug=True)
