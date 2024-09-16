import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests

def fetch_data_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: HTTP {response.status_code}")

def load_and_preprocess_data(data):
    df = pd.DataFrame(data['data'])
    
    # Convert latitude and longitude to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Filter out invalid coordinates
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    
    features = ['latitude', 'longitude', 'water_level_current', 'water_level_normal_level', 
                'water_level_alert_level', 'water_level_warning_level', 'water_level_danger_level']
    
    # Convert water level features to numeric, replacing non-numeric values with NaN
    for feature in features[2:]:  # Skip latitude and longitude
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=features)
    
    # Extract features for clustering
    X = df_clean[features]
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_clean

def perform_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

def visualize_clusters(df, cluster_labels):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('K-means Clustering of Flood Warning Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def analyze_clusters(df, cluster_labels):
    df['Cluster'] = cluster_labels
    cluster_summary = df.groupby('Cluster').agg({
        'water_level_current': 'mean',
        'water_level_normal_level': 'mean',
        'water_level_alert_level': 'mean',
        'water_level_warning_level': 'mean',
        'water_level_danger_level': 'mean'
    })
    return cluster_summary

def main(api_url):
    data = fetch_data_from_api(api_url)
    X_scaled, df_clean = load_and_preprocess_data(data)
    
    if len(df_clean) < 3:
        print("Not enough valid data points for clustering. Please check the data source.")
        return
    
    cluster_labels = perform_kmeans(X_scaled)
    visualize_clusters(df_clean, cluster_labels)
    cluster_summary = analyze_clusters(df_clean, cluster_labels)
    print("Cluster Summary:")
    print(cluster_summary)
    print("\nTotal number of stations:", data['meta']['total'])
    print("Number of stations used in clustering:", len(df_clean))

if __name__ == "__main__":
    api_url = "https://api.data.gov.my/flood-warning/?meta=true"
    main(api_url)