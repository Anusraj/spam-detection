import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class JobFraudDetector:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_features(self, df):
        """Prepare features for training."""
        # Combine relevant text fields
        df['combined_text'] = df['title'] + ' ' + df['description']
        if 'company' in df.columns:
            df['combined_text'] += ' ' + df['company']
        if 'location' in df.columns:
            df['combined_text'] += ' ' + df['location']
            
        # Preprocess text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df
    
    def train(self, df, target_column='is_fraud'):
        """Train the model."""
        # Prepare features
        df = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'],
            df[target_column],
            test_size=0.2,
            random_state=42,
            stratify=df[target_column]
        )
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return f1
    
    def predict(self, df):
        """Make predictions on new data."""
        df = self.prepare_features(df)
        probabilities = self.pipeline.predict_proba(df['processed_text'])
        predictions = self.pipeline.predict(df['processed_text'])
        
        df['fraud_probability'] = probabilities[:, 1]
        df['predicted_fraud'] = predictions
        
        return df
    
    def save_model(self, path='model.joblib'):
        """Save the trained model."""
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load_model(cls, path='model.joblib'):
        """Load a trained model."""
        detector = cls()
        detector.pipeline = joblib.load(path)
        return detector

if __name__ == "__main__":
    # Example usage
    # Load your training data
    # df = pd.read_csv('path_to_your_data.csv')
    # detector = JobFraudDetector()
    # detector.train(df)
    # detector.save_model()
    pass 