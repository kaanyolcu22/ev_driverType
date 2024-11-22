import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EnhancedEVVisualizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def create_advanced_features(self, data):
        df = data.copy()
        
        # Time patterns
        df['Hour'] = pd.to_datetime(df['Charging Start Time']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['Charging Start Time']).dt.dayofweek
        df['Weekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
        
        # Charging time patterns (consistency)
        user_hour_stats = df.groupby('User ID')['Hour'].agg(['std', 'nunique']).fillna(0)
        df = df.merge(user_hour_stats, left_on='User ID', right_index=True, 
                     suffixes=('', '_consistency'))
        
        # Regular timing patterns
        df['Morning_Charge'] = df['Hour'].between(6, 9).astype(int)
        df['Evening_Charge'] = df['Hour'].between(17, 20).astype(int)
        
        # Trip length distributions
        distance_stats = df.groupby('User ID')['Distance Driven (since last charge) (km)'].agg([
            'mean', 'std', 'min', 'max',
            lambda x: stats.skew(x, nan_policy='omit'),
            lambda x: stats.kurtosis(x, nan_policy='omit')
        ]).fillna(0)
        distance_stats.columns = ['avg_distance', 'std_distance', 'min_distance', 
                                'max_distance', 'distance_skew', 'distance_kurtosis']
        df = df.merge(distance_stats, left_on='User ID', right_index=True)
        
        # Energy consumption patterns
        df['Energy_Per_km'] = df['Energy Consumed (kWh)'] / np.maximum(df['Distance Driven (since last charge) (km)'], 0.1)
        df['Charge_Rate'] = df['Energy Consumed (kWh)'] / np.maximum(df['Charging Duration (hours)'], 0.1)
        
        energy_stats = df.groupby('User ID').agg({
            'Energy_Per_km': ['mean', 'std'],
            'Charge_Rate': ['mean', 'std'],
            'State of Charge (Start %)': ['mean', 'std'],
            'State of Charge (End %)': ['mean', 'std']
        }).fillna(0)
        energy_stats.columns = [f"{col[0]}_{col[1]}" for col in energy_stats.columns]
        df = df.merge(energy_stats, left_on='User ID', right_index=True)
        
        # Location diversity
        location_stats = df.groupby('User ID').agg({
            'Charging Station Location': ['nunique', 'count']
        })
        df['Location_Diversity'] = df['User ID'].map(
            location_stats[('Charging Station Location', 'nunique')] / 
            location_stats[('Charging Station Location', 'count')]
        )
        
        # Usage patterns
        df['Deep_Discharge'] = (df['State of Charge (Start %)'] < 20).astype(int)
        df['Full_Charge'] = (df['State of Charge (End %)'] > 90).astype(int)
        df['Quick_Charge'] = (df['Charging Duration (hours)'] < 1).astype(int)
        
        return df
        
    def visualize_data(self, data_path):
        # Load and process data
        data = pd.read_csv(data_path)
        data = self.create_advanced_features(data)
        
        features = [
            # Timing consistency
            'std', 'nunique', 'Morning_Charge', 'Evening_Charge',
            
            # Trip patterns
            'avg_distance', 'std_distance', 'distance_skew', 'distance_kurtosis',
            
            # Energy patterns
            'Energy_Per_km_mean', 'Energy_Per_km_std',
            'Charge_Rate_mean', 'Charge_Rate_std',
            'State of Charge (Start %)_mean', 'State of Charge (End %)_mean',
            
            # Location and behavior
            'Location_Diversity', 'Deep_Discharge', 'Full_Charge', 'Quick_Charge'
        ]
        
        X = data[features]
        y = data['User Type']
        
        # Impute and scale
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=50)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color mapping
        unique_labels = y.unique()
        colors = sns.color_palette("husl", len(unique_labels))
        color_dict = dict(zip(unique_labels, colors))
        
        # PCA plot
        for label, color in color_dict.items():
            mask = y == label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=label, alpha=0.6)
        
        ax1.set_title('PCA Visualization with Enhanced Features')
        ax1.set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
        ax1.legend()
        
        # t-SNE plot
        for label, color in color_dict.items():
            mask = y == label
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], label=label, alpha=0.6)
        
        ax2.set_title('t-SNE Visualization with Enhanced Features')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print feature importance
        feature_importance = pd.DataFrame(
            abs(pca.components_[:2].T),
            columns=['PC1', 'PC2'],
            index=features
        )
        
        print("\nTop 5 Most Important Features for Each Component:")
        print("\nPC1:")
        print(feature_importance.sort_values('PC1', ascending=False).head())
        print("\nPC2:")
        print(feature_importance.sort_values('PC2', ascending=False).head())
        
        total_var = sum(pca.explained_variance_ratio_)
        print(f"\nTotal Variance Explained: {total_var:.2%}")

if __name__ == "__main__":
    visualizer = EnhancedEVVisualizer()
    visualizer.visualize_data('ev_charging_patterns.csv')
