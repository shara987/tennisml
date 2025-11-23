import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt

class PoseDataset(Dataset):
    def __init__(self, json_files, target_frames=100):
        """
        json_files: list of paths to preprocessed JSON sequences (preprocessed with preprocess_sequence)
        """
        self.data = []
        for file in json_files:
            with open(file, "r") as f:
                seq = np.array(json.load(f), dtype=np.float32)
                self.data.append(seq)
        self.data = np.stack(self.data)  # (N, T, J, 2)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return tensor of shape (T, J*2)
        seq = self.data[idx]
        seq = seq.reshape(seq.shape[0], -1)  # (T, 26)
        return torch.tensor(seq, dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(26, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        # x: (batch, T, 26)
        x = x.transpose(1,2)          # (batch, 26, T)
        h = self.conv(x).squeeze(-1)  # (batch, 128)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, T=100):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(latent_dim, 128*T)
        self.deconv = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 26, 3, padding=1)
        )
        
    def forward(self, z):
        h = self.fc(z)               # (batch, 128*T)
        h = h.view(-1, 128, self.T)  # (batch,128,T)
        x = self.deconv(h)            # (batch,26,T)
        x = x.transpose(1,2)          # (batch, T, 26)
        x = x.view(-1, self.T, 13, 2) # (batch, T, J=13, 2)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=32, T=100):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, T)
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar
    
def vae_loss(recon, x, mu, logvar, beta=3):
    recon_loss = ((recon - x.view(x.shape[0], x.shape[1], 13, 2))**2).mean()
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta*KL

# Example: list of JSON files
json_files = ["pose_normalized_1.json", "pose_normalized_2.json", ...]  

dataset = PoseDataset(json_files)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=32, T=100).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 50

for epoch in range(epochs):
    vae.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar, beta=3)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

vae.eval()
with torch.no_grad():
    sample = dataset[0].unsqueeze(0).to(device)  # a single serve
    recon, _, _ = vae(sample)
    
    # Error per joint
    error = torch.abs(recon - sample.view(1,100,13,2)).cpu().numpy()  # (1,100,13,2)
    
    # Sum over x/y for each joint
    joint_error = error.sum(axis=3).mean(axis=1)  # (100,13)
    
    # Example: plot joint error over frames
    plt.figure(figsize=(10,4))
    for j in range(13):
        plt.plot(joint_error[:,j], label=f'Joint {j}')
    plt.xlabel("Frame")
    plt.ylabel("Reconstruction error")
    plt.title("Joint-wise reconstruction error")
    plt.legend()
    plt.show()
