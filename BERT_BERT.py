import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

df = pd.read_csv('extended_videos_with_subtitles.csv')

captions = (
    df.groupby("user_id")["subtitles"]
      .apply(lambda x: " ".join(x)) 
      .reset_index()
)

creators = []
for _, row in captions.iterrows():
    new_row = row['subtitles'].replace(",", " ")
    new_row = " ".join(new_row.split())  # collapse multiple spaces
    creators.append(new_row)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

enc = tokenizer(
    creators,
    padding=True,
    truncation=True,
    max_length=128,          # adjust if you need longer inputs (<=512)
    return_tensors='pt'
)

with torch.no_grad():
    out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
    last_hidden = out.last_hidden_state               # [batch, seq, hidden]

# Mean-pool over tokens (masking out padding)
mask = enc['attention_mask'].unsqueeze(-1)           # [batch, seq, 1]
sum_embeddings = (last_hidden * mask).sum(dim=1)     # [batch, hidden]
lengths = mask.sum(dim=1)                            # [batch, 1]
sentence_embeddings = sum_embeddings / lengths       # [batch, hidden]  -> torch.FloatTensor

# Convert to NumPy for scikit-learn
X = sentence_embeddings.cpu().numpy()                # shape (n_samples, hidden_size)

# Now run K-Means
kmeans = KMeans(n_clusters=8, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print(creators)
print(labels)