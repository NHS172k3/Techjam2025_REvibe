import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans

elmo = hub.load("https://tfhub.dev/google/elmo/3")

def get_elmo_embedding(sentences):
    embeddings = elmo.signatures["default"](tf.constant(sentences))["elmo"]
    return embeddings

sentences = [
    "The bank will approve your loan.",
    "He sat by the bank of the river."
]

embeddings = get_elmo_embedding(sentences)
print(embeddings)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(embeddings)

labels = kmeans.labels_

print(labels)