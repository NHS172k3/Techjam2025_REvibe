import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd

def advanced_clustering(df):
    """Use multiple metrics and validation techniques for optimal clustering"""
    
    # Use a better sentence transformer model
    model = SentenceTransformer('all-mpnet-base-v2')  # Better than distiluse
    
    captions = df['subtitles'].fillna('').astype(str).tolist()
    embeddings = model.encode(captions, normalize_embeddings=True)
    
    # Multiple clustering validation metrics
    def evaluate_clustering(X, k_range=range(3, 15)):
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Multiple validation metrics
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin,
                'inertia': inertia,
                'labels': labels
            })
        
        return results
    
    # Find optimal k using ensemble of metrics
    results = evaluate_clustering(embeddings)
    
    # Normalize scores and combine (higher is better for silhouette/calinski, lower for davies_bouldin)
    normalized_results = []
    for result in results:
        silhouette_norm = (result['silhouette'] - min(r['silhouette'] for r in results)) / \
                         (max(r['silhouette'] for r in results) - min(r['silhouette'] for r in results))
        
        calinski_norm = (result['calinski_harabasz'] - min(r['calinski_harabasz'] for r in results)) / \
                       (max(r['calinski_harabasz'] for r in results) - min(r['calinski_harabasz'] for r in results))
        
        davies_norm = 1 - (result['davies_bouldin'] - min(r['davies_bouldin'] for r in results)) / \
                     (max(r['davies_bouldin'] for r in results) - min(r['davies_bouldin'] for r in results))
        
        # Ensemble score (weighted combination)
        ensemble_score = 0.4 * silhouette_norm + 0.4 * calinski_norm + 0.2 * davies_norm
        
        normalized_results.append({
            **result,
            'ensemble_score': ensemble_score
        })
    
    # Select best k
    best_result = max(normalized_results, key=lambda x: x['ensemble_score'])
    optimal_k = best_result['k']
    
    print(f"Optimal k={optimal_k} (ensemble_score={best_result['ensemble_score']:.3f})")
    
    # Final clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, n_init=25, random_state=42)
    final_labels = final_kmeans.fit_predict(embeddings)
    
    df = df.copy()
    df['cluster'] = final_labels
    
    return df, embeddings, final_kmeans

# ADD THIS NEW FUNCTION - keep your existing function name for compatibility
def sort_category(df):
    """Wrapper function to maintain compatibility with main.py"""
    df_clustered, embeddings, kmeans_model = advanced_clustering(df)
    return df_clustered