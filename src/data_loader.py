import pandas as pd
import os

def load_data(filepath):
    """Loads the dataset from csv."""
    print(f"\n📥 Loading Data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("✅ Data Loaded Successfully.")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File {filepath} not found.")
        return None

def clean_data(df):
    """Enhanced cleaning steps."""
    print("🧹 Cleaning Data...")
    
    # 1. Standardize Job Titles
    if 'title' in df.columns:
        # Lowercase, strip, and remove special characters like markdown artifacts, emojis
        df['clean_title'] = df['title'].str.lower().str.strip()
        # Remove noisy prefixes like !, [, (, and non-alphanumeric at start
        df['clean_title'] = df['clean_title'].str.replace(r'^[^a-zA-Z0-9]+', '', regex=True)
        # Filter out extremely short or garbage titles
        df = df[df['clean_title'].str.len() > 3].copy()
    
    # 2. Clean Work Type
    if 'formatted_work_type' in df.columns:
        df['formatted_work_type'] = df['formatted_work_type'].fillna('Unknown')
        
    return df
