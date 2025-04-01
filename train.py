import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from model import SiaNet
from torch.nn.utils.rnn import pad_sequence

ledger_df = pd.read_csv("ledger_transactions.csv")
bank_df = pd.read_csv("bank_transactions.csv")

ledger_df.head()
bank_df.head()

plt.figure(figsize=(10,5))
sns.histplot(ledger_df['Amount'], bins=30, color='blue', label='Ledger')
sns.histplot(bank_df['Amount'], bins=30, color='orange', label='Bank')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.show()

ledger_df['Description'].value_counts().head(10)
bank_df['Description'].value_counts().head(10)

merged_df = ledger_df.merge(bank_df, on='Description', suffixes=('_ledger', '_bank'))
merged_df['Amount_Diff'] = np.abs(merged_df['Amount_ledger'] - merged_df['Amount_bank'])

merged_df['Date_ledger'] = pd.to_datetime(merged_df['Date_ledger'])
merged_df['Date_bank'] = pd.to_datetime(merged_df['Date_bank'])
merged_df['Date_Diff'] = np.abs((merged_df['Date_ledger'] - merged_df['Date_bank']).dt.days)

merged_df.head()

def collate_fn(batch):
    texts, numerics = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    numerics_tensor = torch.stack(numerics)
    return texts_padded, numerics_tensor

class TransactionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['Description']
        numeric_features = torch.tensor([row['Amount_Diff'], row['Date_Diff']], dtype=torch.float)
        text_encoded = torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long)
        return text_encoded, numeric_features

dataset = TransactionDataset(merged_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
data_size = len(dataset)

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for text, numeric in dataloader:
        batch_size = text.size(0)

        text_pos, num_pos = text.to(device), numeric.to(device)
        indices = torch.randperm(data_size)[:batch_size]
        text_neg, num_neg = zip(*[dataset[int(i)] for i in indices])
        text_neg_padded = pad_sequence(text_neg, batch_first=True, padding_value=0).to(device)
        nnt = torch.stack(num_neg).to(device)

        optimizer.zero_grad()

        # Forward pass (positive pairs)
        out1_pos, out2_pos = model(text_pos, num_pos, text_pos, num_pos)
        dist_pos = F.pairwise_distance(out1_pos, out2_pos)

        # Forward pass (negative pairs)
        out1_neg, out2_neg = model(text_pos, num_pos, text_neg_padded, nnt)
        distance_neg = F.pairwise_distance(out1_neg, out2_neg)

        # Create labels
        labels_pos = torch.ones(batch_size).to(device)
        labels_neg = torch.zeros(batch_size).to(device)

        # Define contrastive loss
        margin = 1.0
        p_loss = torch.mean(labels_pos * dist_pos ** 2)
        n_loss = torch.mean((1 - labels_neg) * F.relu(margin - distance_neg) ** 2)
        loss = p_loss + n_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

from torch.nn.utils.rnn import pad_sequence

def encode_text(texts):
    return [torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long) for text in texts]

unmatched_df = ledger_df.sample(100).reset_index(drop=True)
unmatched_df2 = bank_df.sample(100).reset_index(drop=True)

unmatched_df['Amount_Diff'] = np.abs(unmatched_df['Amount'] - unmatched_df2['Amount'])
unmatched_df['Date_Diff'] = np.abs((pd.to_datetime(unmatched_df['Date']) - pd.to_datetime(unmatched_df2['Date'])).dt.days)

text1_unmatched_encoded = encode_text(unmatched_df['Description'])
text2_unmatched_encoded = encode_text(unmatched_df2['Description'])

text1_unmatched_padded = pad_sequence(text1_unmatched_encoded, batch_first=True, padding_value=0).to(device)
text2_unmatched_padded = pad_sequence(text2_unmatched_encoded, batch_first=True, padding_value=0).to(device)

numeric_unmatched_tensor = torch.tensor(unmatched_df[['Amount_Diff', 'Date_Diff']].values, dtype=torch.float).to(device)
numeric_unmatched_tensor2 = torch.tensor(unmatched_df[['Amount_Diff', 'Date_Diff']].values, dtype=torch.float).to(device)

# get distances for unmatched pairs
model.eval()
with torch.no_grad():
    out1_unmatched, out2_unmatched = model(text1_unmatched_padded, numeric_unmatched_tensor,
                                           text2_unmatched_padded, numeric_unmatched_tensor2)
    distance_unmatched = F.pairwise_distance(out1_unmatched, out2_unmatched)

# get distances for matched pairs 
matched_text_encoded = encode_text(merged_df['Description'])
matched_text_padded = pad_sequence(matched_text_encoded, batch_first=True, padding_value=0).to(device)
numeric_matched_tensor = torch.tensor(merged_df[['Amount_Diff', 'Date_Diff']].values, dtype=torch.float).to(device)

with torch.no_grad():
    out1_matched, out2_matched = model(matched_text_padded, numeric_matched_tensor,
                                       matched_text_padded, numeric_matched_tensor)
    distance_matched = F.pairwise_distance(out1_matched, out2_matched)

# graph it
matched_distances = distance_matched.cpu().numpy()
unmatched_distances = distance_unmatched.cpu().numpy()

torch.save(model.state_dict(), 'siamese_model.pth')

plt.figure(figsize=(10,5))
plt.hist(matched_distances, alpha=0.6, bins=20, label='Matched pairs')
plt.hist(unmatched_distances, alpha=0.6, bins=20, label='Unmatched pairs')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distance Distribution: Matched vs Unmatched')
plt.show()

