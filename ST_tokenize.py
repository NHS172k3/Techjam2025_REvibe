import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

model = SentenceTransformer('distiluse-base-multilingual-cased-v2') 

def sort_category(df):
    captions = (
        df['subtitles']     
        .fillna('')                 
        .astype(str)           
        .tolist()                
    )

    emb = model.encode(captions, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    def best_k(X, k_range=range(5, 11)):
        best_k, best_score = None, -1
        for k in k_range:
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            score = silhouette_score(X, labels, metric='euclidean')
            if score > best_score:
                best_k, best_score = k, score
        return best_k, best_score

    k, score = best_k(emb, range(5, 11))
    print(f"Chosen k={k} (silhouette={score:.3f})")

    kmeans = KMeans(n_clusters=k, n_init=25, random_state=42)
    labels = kmeans.fit_predict(emb)
    pd.set_option('display.max_rows', None)

    df = df.copy()
    df['cluster'] = labels
    print(df)

    return df