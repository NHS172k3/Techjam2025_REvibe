import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans

elmo = hub.load("https://tfhub.dev/google/elmo/3")

df = pd.read_csv('mock_videos.csv')

user_tags = (
    df.groupby("user_id")["tags"]
      .apply(lambda x: " ".join(x)) 
      .reset_index()
)

print(user_tags)

creators = []
for _, row in user_tags.iterrows():
    new_row = row['tags'].replace(",", " ")
    new_row = " ".join(new_row.split())  # collapse multiple spaces
    creators.append(new_row)

print(creators) 

def get_elmo_embedding(sentences):
    # sentences: list[str] with shape [batch]
    out = elmo.signatures["default"](tf.constant(sentences))
    return out["elmo"]  # [batch, max_len, 1024]

# 1) Get token-level embeddings
vectors = get_elmo_embedding(creators)        # shape [batch, max_len, 1024]

# 2) Mean-pool over tokens -> 2D matrix for KMeans
X = tf.reduce_mean(vectors, axis=1).numpy()   # shape [batch, 1024]
print("Embedding shape for KMeans:", X.shape)

# 3) KMeans
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print("Cluster labels:", labels)

# (optional) attach back to user_ids so you know who is who
user_tags["cluster"] = labels
print(user_tags)