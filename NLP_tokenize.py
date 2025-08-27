import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

df = pd.read_csv('mock_videos.csv')

user_tags = (
    df.groupby("user_id")["tags"]
      .apply(lambda x: " ".join(x)) 
      .reset_index()
)

creators = {}

for index, row in user_tags.iterrows():
    creators[row['user_id']] = row['tags']

docs = [" ".join(tags) for tags in creators.values()]

print(docs)

model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(docs) 

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_

print(labels)