# HR Analytics & Employee Attrition Prediction System

A complete Machine Learning and Flask-based web application for analyzing human resource metrics, forecasting employee attrition risk, and delivering actionable retention insights.

## Overview

This project analyzes the IBM HR Analytics Employee Attrition dataset to identify factors driving employee turnover and provides:
- **Interactive Web Dashboard**: Executive overview, KPI scorecards, risk explorer, and retention scenario simulator.
- **Attrition Risk Predictor**: Real-time prediction tool based on machine learning models.
- **Exploratory Data Analysis**: Jupyter notebooks detailing data preparation, exploratory visualization, statistical testing, and model comparison.
- **Model Evaluation**: Comparison between Logistic Regression, Random Forest, and Gradient Boosting classifiers.

---

## Project Structure

```text
├── CODE/
│   ├── 01_Data_Preparation.ipynb       # Data cleaning & preprocessing
│   ├── 02_Data_Modeling.ipynb          # Model training & benchmarking
│   ├── 03_Data_Visualization.ipynb     # Exploratory visual analysis
│   └── 04_Statistical-Analysis.ipynb   # Statistical significance tests
├── data/                               # Cleaned dataset for the web application
├── DATASET/                            # Raw dataset
├── models/
│   ├── best_model.pkl                  # Serialized best-performing ML model
│   ├── feature_columns.pkl             # Encoded feature column list
│   └── metrics.json                    # Model evaluation metrics & ROC curves
├── static/
│   ├── css/                            # Dashboard styles
│   └── js/                             # Interactive charts & client-side logic
├── templates/                          # Jinja2 HTML templates
├── utils/
│   └── preprocess.py                   # Feature transformation helpers
├── app.py                              # Flask application backend
├── train_model.py                      # Standalone training script
└── requirements.txt                    # Project dependencies
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/PriyaMeena12/HR-analytic-system.git
cd HR-analytic-system
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)
To retrain the ML models and refresh saved weights:
```bash
python train_model.py
```

### 5. Run the Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Analysis**: Jupyter Notebooks, Seaborn, Matplotlib

---

## License
This project is open-source and available under the [MIT License](LICENSE).
