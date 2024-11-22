import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

class EVUserClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_transformer = None
        self.model = None
        
    def create_features(self, data):
        """Create enhanced feature set based on analysis insights"""
        df = data.copy()
        
        # Basic derived features
        df['Energy_Per_km'] = df['Energy Consumed (kWh)'] / df['Distance Driven (since last charge) (km)']
        df['Cost_Per_km'] = df['Charging Cost (USD)'] / df['Distance Driven (since last charge) (km)']
        
        # Time-based features
        df['Charging_Start_Time'] = pd.to_datetime(df['Charging Start Time'])
        df['Hour'] = df['Charging_Start_Time'].dt.hour
        df['DayOfWeek'] = df['Charging_Start_Time'].dt.dayofweek
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Handle time of day as categorical
        tod_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
        df['Time_Of_Day_Code'] = df['Time of Day'].map(tod_map)
        
        # Charging efficiency features
        df['Charging_Speed'] = df['Energy Consumed (kWh)'] / df['Charging Duration (hours)']
        df['SOC_Change'] = df['State of Charge (End %)'] - df['State of Charge (Start %)']
        df['SOC_Rate'] = df['SOC_Change'] / df['Charging Duration (hours)']
        
        # Temperature interaction features
        df['Temp_Efficiency'] = df['Energy_Per_km'] * np.exp(-df['Temperature (°C)']/30)
        df['Temp_Category'] = pd.cut(
            df['Temperature (°C)'], 
            bins=[-np.inf, 0, 15, 25, np.inf],
            labels=['Cold', 'Cool', 'Moderate', 'Warm']
        )
        
        # Battery and vehicle features
        df['Battery_Utilization'] = df['Energy Consumed (kWh)'] / df['Battery Capacity (kWh)']
        df['Age_Category'] = pd.cut(
            df['Vehicle Age (years)'],
            bins=[-np.inf, 2, 4, 6, np.inf],
            labels=['New', 'Mid', 'Mature', 'Older']
        )
        
        # Clean up infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    
    def prepare_features(self, df):
        """Select and prepare features for modeling"""
        feature_columns = [
            # Time features
            'Hour_Sin', 'Hour_Cos', 'Weekend', 'Time_Of_Day_Code',
            
            # Core charging metrics
            'Energy Consumed (kWh)', 'Charging Duration (hours)', 
            'Charging Rate (kW)', 'Distance Driven (since last charge) (km)',
            
            # Efficiency metrics
            'Energy_Per_km', 'Cost_Per_km', 
            'Charging_Speed', 'Battery_Utilization',
            
            # State of charge features
            'State of Charge (Start %)', 'State of Charge (End %)',
            'SOC_Change', 'SOC_Rate',
            
            # Environmental and vehicle features
            'Temperature (°C)', 'Vehicle Age (years)'
        ]
        
        return df[feature_columns]
    
    def fit(self, X, y):
        """Fit the model with feature selection and preprocessing"""
        # Create enhanced features
        X_enhanced = self.create_features(X)
        X_features = self.prepare_features(X_enhanced)
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create preprocessing pipeline
        self.feature_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                max_features=12
            ))
        ])
        
        # Create and train the final model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric=['mlogloss', 'merror'],  # Moved eval_metric here
            objective='multi:softmax',
            num_class=len(np.unique(y_encoded)),
            use_label_encoder=False  # Added to prevent warning
        )
        
        # Fit the pipeline
        X_transformed = self.feature_transformer.fit_transform(X_features, y_encoded)
        
        print("\nTraining XGBoost model...")
        # Fit without eval_metric parameter
        self.model.fit(
            X_transformed, 
            y_encoded,
            verbose=True
        )
        
        # Print feature importance
        feature_names = X_features.columns
        selected_features_mask = self.feature_transformer.named_steps['feature_selection'].get_support()
        selected_features = feature_names[selected_features_mask]
        
        print("\nSelected Features:")
        for feature in selected_features:
            print(f"- {feature}")
            
        # Print feature importances
        importances = pd.DataFrame({
            'feature': selected_features,
            'importance': self.model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances)
        
        return self

    def predict(self, X):
        """Predict user types"""
        try:
            X_enhanced = self.create_features(X)
            X_features = self.prepare_features(X_enhanced)
            X_transformed = self.feature_transformer.transform(X_features)
            predictions = self.model.predict(X_transformed)
            return self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        try:
            print("\nEvaluating model performance...")
            
            X_enhanced = self.create_features(X)
            X_features = self.prepare_features(X_enhanced)
            y_encoded = self.label_encoder.transform(y)
            
            # Cross-validation
            X_transformed = self.feature_transformer.transform(X_features)
            cv_scores = cross_val_score(self.model, X_transformed, y_encoded, cv=5)
            
            print("\nCross-validation Scores:")
            print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Detailed classification report
            predictions = self.predict(X)
            print("\nClassification Report:")
            print(classification_report(y, predictions))
            
            # Confusion Matrix
            print("\nConfusion Matrix:")
            conf_matrix = confusion_matrix(y, predictions)
            print(conf_matrix)
            
            # Print confusion matrix with labels
            user_types = self.label_encoder.classes_
            print("\nConfusion Matrix with Labels:")
            conf_matrix_df = pd.DataFrame(
                conf_matrix,
                index=[f'True_{t}' for t in user_types],
                columns=[f'Pred_{t}' for t in user_types]
            )
            print(conf_matrix_df)
            
            return cv_scores
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        print("Loading data...")
        # Load data
        data = pd.read_csv('../ev_charging_patterns.csv')
        
        print("\nData shape:", data.shape)
        print("\nFeatures available:", list(data.columns))
        
        # Split features and target
        X = data.drop('User Type', axis=1)
        y = data['User Type']
        
        print("\nClass distribution:")
        print(y.value_counts(normalize=True))
        
        print("\nSplitting data into train and test sets...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        print("\nTraining model...")
        classifier = EVUserClassifier()
        classifier.fit(X_train, y_train)
        
        # Evaluate
        classifier.evaluate(X_test, y_test)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise