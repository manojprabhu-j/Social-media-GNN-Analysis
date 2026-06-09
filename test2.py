import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:/Users/manoj/Downloads/user_interactions.csv")

# Create a directed graph
G = nx.DiGraph()

# Add edges from dataset
for _, row in df.iterrows():
    user1, user2 = row['user_id_1'], row['user_id_2']
    
    # Add edge weight (interaction count)
    if G.has_edge(user1, user2):
        G[user1][user2]['weight'] += 1
    else:
        G.add_edge(user1, user2, weight=1)


import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

# Convert node labels to numerical indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}
edges_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# Create Example Graph
G = nx.karate_club_graph()  # Load a sample social network

# Convert to PyG Format
data = from_networkx(G)

# Train Node2Vec for Node Embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node2vec =Node2Vec(data.edge_index, embedding_dim=128, walk_length=10, context_size=5, walks_per_node=10)
node2vec =node2vec.to(device)
from torch_geometric.nn import Node2Vec
import torch
import torch.optim as optim

# Check and adjust num_nodes to prevent out-of-bound errors:
num_nodes = data.num_nodes
if data.edge_index.max() >= num_nodes:
    print("Warning: Adjusting num_nodes due to out-of-bound indices!")
    num_nodes = data.edge_index.max().item() + 1  # Adjust to max index + 1

# Initialize Node2Vec with adjusted num_nodes
node2vec = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10, context_size=5,
                    walks_per_node=10, num_nodes=num_nodes)

# Define optimizer
optimizer = optim.Adam(node2vec.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Sample random walks
    pos_rw, neg_rw = node2vec.sample(128)

    # Check if any walk contains invalid indices
    if pos_rw.max() >= node2vec.num_nodes or neg_rw.max() >= node2vec.num_nodes:
        print("Error: Random walks contain out-of-bound indices!")
        print(f"Max index in pos_rw: {pos_rw.max().item()}, Max index in neg_rw: {neg_rw.max().item()}")
        break

    # Calculate loss and backpropagate
    loss = node2vec.loss(pos_rw, neg_rw)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training completed.")




embeddings = node2vec().cpu().detach().numpy()  # Get learned node embeddings

# Map Node IDs and Create Edge Index
node_mapping = {node: i for i, node in enumerate(G.nodes())}
edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# Convert Node Features (Embeddings) to Tensor
node_features = torch.tensor([embeddings[node_mapping[node]] for node in G.nodes()], dtype=torch.float)

# Assign Labels (Influencer = 1 if Degree > 2, else 0)
labels = torch.tensor([1 if G.degree[node] > 2 else 0 for node in G.nodes()], dtype=torch.long)

# Create PyG Graph Data
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Check Data Format
print(data)
import numpy as np
import torch

# Debugging: Check the first 10 nodes and array shape
print(f"Nodes in the graph: {list(G.nodes())[:10]}")
print(f"Embeddings shape: {embeddings.shape}")

# Handle integer-based access
node_features_list = [
    embeddings[node] if node < embeddings.shape[0] else np.zeros(128) for node in G.nodes()
]

# Efficiently convert to tensor
node_features_array = np.array(node_features_list)
node_features = torch.tensor(node_features_array, dtype=torch.float)

# Debugging: Check tensor shape after creation
print(f"Node features tensor shape: {node_features.shape}")


# Dummy labels (Influencer = 1, Non-Influencer = 0)
labels = torch.tensor([1 if G.degree[node] > 2 else 0 for node in G.nodes()], dtype=torch.long)

# Create Graph Data
data = Data(x=node_features, edge_index=edges_index, y=labels)
# Draw Graph
plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
plt.title("Social Media Interaction Graph")
plt.show()

from node2vec import Node2Vec

# Train Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Convert embeddings to dictionary
embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}

# Convert to DataFrame
embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
embedding_df.to_csv("user_embeddings.csv")

import torch
from torch_geometric.data import Data

# Convert node labels to numerical indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}
edges_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

# Convert embeddings to tensor
node_features = torch.tensor([embeddings[str(node)] for node in G.nodes()], dtype=torch.float)

# Dummy labels (Influencer = 1, Non-Influencer = 0)
labels = torch.tensor([1 if G.degree[node] > 2 else 0 for node in G.nodes()], dtype=torch.long)

# Create Graph Data
data = Data(x=node_features, edge_index=edges_index, y=labels)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train Model
model = GNN(input_dim=64, hidden_dim=32, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

print("Model Trained Successfully!")

from sklearn.metrics import accuracy_score

# Predict labels
model.eval()
predictions = model(data.x, data.edge_index).argmax(dim=1)

# Evaluate Accuracy
accuracy = accuracy_score(data.y.numpy(), predictions.numpy())
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import networkx as nx
import matplotlib.pyplot as plt
import community  # Python-Louvain for community detection

# Detect communities
partition = community.best_partition(G)
colors = [partition[node] for node in G.nodes()]

# Draw graph with colors
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.jet, node_size=800)
plt.title("Social Media Network with Community Detection")
plt.show()

import numpy as np

# Assign colors based on interaction strength
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
edge_colors = ['blue' if w < 3 else 'yellow' if w < 6 else 'red' for w in edge_weights]

# Draw graph with colored edges
plt.figure(figsize=(8, 8))
nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=800)
plt.title("Social Media Graph with Interaction Strength Colors")
plt.show()

pagerank = nx.pagerank(G)
node_sizes = [5000 * pagerank[node] for node in G.nodes()]  # Scale node size
node_colors = ['red' if pagerank[node] > 0.05 else 'blue' for node in G.nodes()]

plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True)
plt.title("Social Media Graph with Influencer Highlighting")
plt.show()

