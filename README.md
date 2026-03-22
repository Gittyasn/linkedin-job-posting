# 🚀 LinkedIn Job Market Insights Dashboard

## 📝 Project Overview
This is a **Major Data Science Project** featuring an interactive **Streamlit Web Dashboard**. It analyzes thousands of LinkedIn job postings to provide real-time insights, salary predictions, and skill extraction.

![Streamlit](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png)

## 🌟 Key Features
### 1. 📊 Interactive Market Analysis (EDA)
- **Dynamic Charts**: Filter job titles and locations on the fly.
- **Visualizations**: Salaries, Work Types, and Top Hiring Companies.

### 2. 🤖 ML Salary Predictor
- **Real-Time Prediction**: Input a job title, location, and work type to get an instant salary estimate.
- **Model**: optimized Random Forest Regressor (R² Score displayed live).

### 3. 🧠 NLP Skill Extraction
- **Word Clouds**: Visualizes the most in-demand skills from job descriptions.

## 📂 Project Structure
```
├── app.py                 # <--- Main Dashboard App
├── run_app.py            # <--- Launcher Script
├── requirements.txt      # Dependencies (includes streamlit)
├── postings.csv          # Dataset
└── src/                  # Backend Modules
    ├── data_loader.py    # Data Processing
    ├── eda_plots.py      # Visualization Logic
    ├── ml_models.py      # Machine Learning
    └── nlp_analysis.py   # NLP & WordCloud
```

## 🚀 How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch the Dashboard**:
    Since sometimes `streamlit` isn't added to path, use this reliable command:
    ```bash
    python run_app.py
    ```
    *The app will open in your browser automatically.*

