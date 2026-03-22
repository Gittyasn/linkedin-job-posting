try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib
except ImportError:
    train_test_split = None
    joblib = None

import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def train_salary_model(df):
    """Trains model and calculates feature importance."""
    if train_test_split is None:
        return None, None, {}

    target_col = 'normalized_salary'
    feature_cols = ['clean_title', 'location', 'formatted_work_type']
    
    # 1. Cleaning
    ml_df = df.dropna(subset=feature_cols + [target_col]).copy()
    ml_df = ml_df[(ml_df[target_col] > 10000) & (ml_df[target_col] < 500000)]

    if len(ml_df) < 100:
        return None, None, {}

    # 2. Encode
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le

    X = ml_df[feature_cols]
    y = ml_df[target_col]

    # 3. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # 5. Explainable AI: Feature Importance
    importance = model.feature_importances_
    feat_importance = pd.DataFrame({'feature': feature_cols, 'importance': importance}).sort_values('importance', ascending=False)
    
    metrics = {
        "mae": mae, 
        "r2": r2, 
        "samples": len(ml_df),
        "importance": feat_importance # Save importance for the dashboard
    }
    
    # 6. Save
    if joblib:
        try:
            joblib.dump({'model': model, 'encoders': encoders, 'metrics': metrics}, MODEL_PATH)
        except Exception:
            pass

    return model, encoders, metrics

def load_saved_model():
    if joblib and os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            return data['model'], data['encoders'], data['metrics']
        except Exception:
            return None, None, None
    return None, None, None

def predict_salary(model, encoders, title, location, work_type):
    try:
        # Create a DataFrame for prediction to maintain feature names and avoid warnings
        input_data = pd.DataFrame([{
            'clean_title': title.lower().strip(),
            'location': location,
            'formatted_work_type': work_type
        }])
        
        # Encode using the stored encoders
        for col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col].astype(str))
            
        return model.predict(input_data)[0]
    except Exception:
        return None
