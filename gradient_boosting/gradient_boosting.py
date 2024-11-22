import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import logging

class BalancedEVClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_user_patterns(self, df):
        patterns = {}
        for user_id in df['User ID'].unique():
            user_data = df[df['User ID'] == user_id]
            patterns[user_id] = {
                'avg_distance': user_data['Distance Driven (since last charge) (km)'].mean(),
                'avg_duration': user_data['Charging Duration (hours)'].mean(),
                'peak_charging': user_data['Hour'].between(9, 17).mean(),
                'night_charging': user_data['Hour'].between(22, 6).mean(),
                'weekend_ratio': user_data['Weekend'].mean(),
                'low_battery_freq': (user_data['State of Charge (Start %)'] < 20).mean(),
                'full_charges': (user_data['State of Charge (End %)'] > 90).mean(),
                'charge_count': len(user_data)
            }
        return pd.DataFrame.from_dict(patterns, orient='index')
    
    def engineer_features(self, data):
        df = data.copy()
        
        # Time features
        df['Hour'] = pd.to_datetime(df['Charging Start Time']).dt.hour
        df['Weekend'] = pd.to_datetime(df['Charging Start Time']).dt.dayofweek.isin([5,6]).astype(int)
        
        # Extract user patterns
        patterns = self.extract_user_patterns(df)
        
        # Merge patterns back
        df = df.merge(patterns, left_on='User ID', right_index=True, suffixes=('', '_pattern'))
        
        # Derived features
        df['Distance_Per_Hour'] = df['Distance Driven (since last charge) (km)'] / np.maximum(df['Charging Duration (hours)'], 0.1)
        df['Energy_Efficiency'] = df['Distance Driven (since last charge) (km)'] / np.maximum(df['Energy Consumed (kWh)'], 0.1)
        df['Cost_Per_km'] = df['Charging Cost (USD)'] / np.maximum(df['Distance Driven (since last charge) (km)'], 0.1)
        
        return df
    
    def train(self, data_path):
        logging.basicConfig(level=logging.INFO)
        
        # Load and process
        data = pd.read_csv(data_path)
        data = self.engineer_features(data)
        
        features = [
            'avg_distance', 'avg_duration', 'peak_charging',
            'night_charging', 'weekend_ratio', 'low_battery_freq',
            'full_charges', 'charge_count', 'Distance_Per_Hour',
            'Energy_Efficiency', 'Cost_Per_km'
        ]
        
        X = data[features]
        y = self.label_encoder.fit_transform(data['User Type'])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Create sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X_scaled, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train without `class_weights` but using `sample_weights`
        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.01,
            max_iter=1000,
            min_samples_leaf=10,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        
        model.fit(X_train, y_train, sample_weight=sw_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        logging.info(f"Training accuracy: {model.score(X_train, y_train):.4f}")
        logging.info(f"Test accuracy: {model.score(X_test, y_test):.4f}")
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_test, y_pred))
        
        return model


if __name__ == "__main__":
    classifier = BalancedEVClassifier()
    model = classifier.train('../ev_charging_patterns.csv')