import pandas as pd
import numpy as np
from typing import List, Dict, Any

def validate_csv_structure(df: pd.DataFrame) -> bool:
    """
    Validate that the CSV file has the required structure.
    
    Args:
        df: Input DataFrame
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = ['title', 'description']
    optional_columns = ['company', 'location']
    
    # Check required columns
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Check for empty values in required columns
    if df[required_columns].isnull().any().any():
        return False
    
    return True

def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess job posting data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Fill missing values in optional columns
    optional_columns = ['company', 'location']
    for col in optional_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Remove duplicate entries
    df_clean = df_clean.drop_duplicates()
    
    # Remove entries with empty strings
    for col in ['title', 'description']:
        df_clean = df_clean[df_clean[col].str.strip() != '']
    
    return df_clean

def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate sample job posting data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample job posting data
    """
    # Sample job titles
    titles = [
        "Software Engineer", "Data Scientist", "Product Manager",
        "Marketing Specialist", "Sales Representative", "Customer Service",
        "Financial Analyst", "HR Manager", "Operations Director"
    ]
    
    # Sample descriptions
    descriptions = [
        "Join our dynamic team and work on cutting-edge projects.",
        "Looking for experienced professionals to join our growing company.",
        "Work from home opportunity with flexible hours.",
        "Immediate hiring for multiple positions.",
        "No experience required, we provide training."
    ]
    
    # Sample companies
    companies = [
        "TechCorp", "DataSystems", "Global Solutions",
        "Innovation Labs", "Future Enterprises"
    ]
    
    # Sample locations
    locations = [
        "New York, NY", "San Francisco, CA", "Remote",
        "Chicago, IL", "Austin, TX"
    ]
    
    # Generate random data
    data = {
        'title': np.random.choice(titles, n_samples),
        'description': np.random.choice(descriptions, n_samples),
        'company': np.random.choice(companies, n_samples),
        'location': np.random.choice(locations, n_samples)
    }
    
    return pd.DataFrame(data)

def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save analysis results to CSV.
    
    Args:
        df: DataFrame with results
        output_path: Path to save the results
    """
    df.to_csv(output_path, index=False) 