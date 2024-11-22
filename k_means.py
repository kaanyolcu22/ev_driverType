import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

def plot_kmeans_clusters():
    # Load data
    data = pd.read_csv('../ev_charging_patterns.csv')
    
    # Process temporal features
    data['Hour'] = pd.to_datetime(data['Charging Start Time']).dt.hour
    data['Month'] = pd.to_datetime(data['Charging Start Time']).dt.month
    data['DayOfWeek'] = pd.to_datetime(data['Charging Start Time']).dt.dayofweek
    
    # Engineer features
    data['Energy_Per_Hour'] = data['Energy Consumed (kWh)'] / np.maximum(data['Charging Duration (hours)'], 0.1)
    data['Charge_Gained'] = data['State of Charge (End %)'] - data['State of Charge (Start %)']
    data['Battery_Usage'] = data['Energy Consumed (kWh)'] / data['Battery Capacity (kWh)']
    
    features = [
        'Energy_Per_Hour',
        'Charge_Gained',
        'Battery_Usage',
        'Charging Rate (kW)',
        'Charging Cost (USD)',
        'Distance Driven (since last charge) (km)',
        'Temperature (°C)',
        'Vehicle Age (years)',
        'Hour',
        'Month',
        'DayOfWeek'
    ]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(data[features])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap='viridis',
        alpha=0.6
    )
    
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c='red', marker='x', s=200,
        linewidths=3, label='Centroids'
    )
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('EV User Clusters (PCA Visualization)')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    
    # Add user type distribution
    for i in range(3):
        cluster_users = data.loc[clusters == i, 'User Type'].value_counts()
        print(f"\nCluster {i} composition:")
        print(cluster_users)
    
    # Print explained variance
    var_explained = pca.explained_variance_ratio_
    print(f"\nVariance explained: {sum(var_explained):.2%}")
    
    plt.show()


    # Evaluate current clustering
    print("KMeans Silhouette Score:", silhouette_score(X, kmeans.labels_))
    print("KMeans Calinski-Harabasz Index:", calinski_harabasz_score(X, kmeans.labels_))

    # Option 1: Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(X)
    print("Spectral Clustering Silhouette Score:", silhouette_score(X, spectral_labels))

    # Option 2: Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    print("Gaussian Mixture Silhouette Score:", silhouette_score(X, gmm_labels))
    
    return kmeans, pca, clusters

if __name__ == "__main__":
    kmeans, pca, clusters = plot_kmeans_clusters()
