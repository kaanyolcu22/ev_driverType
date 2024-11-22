import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class EVVisualizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def engineer_features(self, data):
        df = data.copy()
        
        # Basic features
        df['Hour'] = pd.to_datetime(df['Charging Start Time']).dt.hour
        df['Weekend'] = pd.to_datetime(df['Charging Start Time']).dt.dayofweek.isin([5,6]).astype(int)
        
        # Efficiency metrics with safe division
        df['Distance_Per_Hour'] = df['Distance Driven (since last charge) (km)'] / np.maximum(df['Charging Duration (hours)'], 0.1)
        df['Energy_Efficiency'] = df['Distance Driven (since last charge) (km)'] / np.maximum(df['Energy Consumed (kWh)'], 0.1)
        df['Cost_Per_km'] = df['Charging Cost (USD)'] / np.maximum(df['Distance Driven (since last charge) (km)'], 0.1)
        
        return df
    
    def visualize_data(self, data_path):
        # Load and process data
        data = pd.read_csv(data_path)
        data = self.engineer_features(data)
        
        features = [
            'Distance_Per_Hour', 'Energy_Efficiency', 'Cost_Per_km',
            'Charging Duration (hours)', 'Charging Rate (kW)',
            'State of Charge (Start %)', 'State of Charge (End %)',
            'Distance Driven (since last charge) (km)', 'Temperature (°C)',
            'Vehicle Age (years)', 'Hour', 'Weekend'
        ]
        
        X = data[features]
        y = data['User Type']
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color mapping
        unique_labels = y.unique()
        color_map = dict(zip(unique_labels, sns.color_palette("husl", len(unique_labels))))
        colors = [color_map[label] for label in y]
        
        # Plot PCA
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
        ax1.set_title('PCA Visualization of EV Charging Patterns')
        ax1.set_xlabel(f'First Principal Component\nExplained Variance: {pca.explained_variance_ratio_[0]:.2%}')
        ax1.set_ylabel(f'Second Principal Component\nExplained Variance: {pca.explained_variance_ratio_[1]:.2%}')
        
        # Plot t-SNE
        scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6)
        ax2.set_title('t-SNE Visualization of EV Charging Patterns')
        ax2.set_xlabel('First t-SNE Component')
        ax2.set_ylabel('Second t-SNE Component')
        
        # Add legends
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color_map[label], 
                                    label=label, markersize=10)
                         for label in unique_labels]
        
        ax1.legend(handles=legend_elements, title='User Types')
        ax2.legend(handles=legend_elements, title='User Types')
        
        plt.tight_layout()
        plt.show()
        
        # Print feature contributions
        feature_importance = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=features
        ).abs()
        
        print("\nTop Feature Contributions to Principal Components:")
        print("\nPC1 contributions:")
        print(feature_importance.sort_values('PC1', ascending=False)['PC1'].head())
        print("\nPC2 contributions:")
        print(feature_importance.sort_values('PC2', ascending=False)['PC2'].head())
        
        # Calculate total variance explained
        total_var = pca.explained_variance_ratio_.sum()
        print(f"\nTotal variance explained by first two components: {total_var:.2%}")

if __name__ == "__main__":
    visualizer = EVVisualizer()
    visualizer.visualize_data('ev_charging_patterns.csv')
