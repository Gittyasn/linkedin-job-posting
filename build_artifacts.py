import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_data, clean_data  # type: ignore
from ml_models import train_salary_model  # type: ignore

def build():
    print("[Build] Building Project Artifacts...")
    
    # 1. Load Data
    df = load_data('postings.csv')
    if df is None:
        print("[Error] Data load failed.")
        return

    # 2. Clean Data
    df = clean_data(df)

    # 3. Train & Save Model
    print("[Model] Training and Saving ML Model...")
    model, encoders, metrics = train_salary_model(df)
    
    if model:
        print(f"[Success] Build Complete! Model saved to src/model.pkl")
        print(f"   MAE: ${metrics['mae']:,.2f}")
        print(f"   R²: {metrics['r2']:.4f}")
    else:
        print("[Error] Model training failed.")

if __name__ == "__main__":
    build()
