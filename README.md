# Social-media-GNN-Analysis
# Social Media Network Analysis Using Node2Vec and Graph Neural Networks

## Overview

This project performs Social Media Network Analysis using Graph Neural Networks (GNNs), Node2Vec embeddings, and NetworkX. The goal is to analyze user interactions, identify influential users, detect communities, and classify users based on their network behavior.

The project combines graph analytics and machine learning techniques to extract meaningful insights from social network interaction data.

---

## Features

* Construction of social interaction graphs from user interaction datasets
* Node embedding generation using Node2Vec
* User classification using Graph Neural Networks (GCN)
* Community detection using the Louvain algorithm
* Influencer identification using PageRank
* Interaction strength visualization
* Social network graph visualization

---

## Technologies Used

* Python
* Pandas
* NetworkX
* PyTorch
* PyTorch Geometric
* Node2Vec
* Matplotlib
* NumPy
* Scikit-learn

---

## Dataset

The dataset contains user interaction information:

| Column      | Description                   |
| ----------- | ----------------------------- |
| user_id_1   | Source user                   |
| user_id_2   | Target user                   |
| interaction | User interaction relationship |

The graph is constructed where:

* Nodes represent users.
* Edges represent interactions between users.
* Edge weights represent interaction frequency.

---

## Project Workflow

### 1. Data Loading

Load social interaction data from CSV files and preprocess the network structure.

### 2. Graph Construction

Create a directed graph using NetworkX where:

* Users are represented as nodes.
* Interactions are represented as directed edges.
* Edge weights capture interaction counts.

### 3. Node Embedding Generation

Node2Vec is used to generate dense vector representations of users.

Benefits:

* Captures structural similarities
* Preserves graph neighborhood information
* Generates meaningful node features for machine learning

### 4. Graph Neural Network (GCN)

A Graph Convolutional Network is trained to classify users.

Architecture:

* Input Layer: Node Embeddings
* Hidden Layer: Graph Convolution
* Output Layer: User Classification

### 5. Community Detection

The Louvain Community Detection algorithm is applied to identify groups of closely connected users.

Outputs:

* Community assignments
* Community visualization

### 6. Influencer Detection

PageRank is used to identify influential users within the network.

Metrics:

* Node importance score
* Influencer highlighting
* Network centrality analysis

---

## Visualizations

### Social Network Graph

Displays user interactions as a directed network.

### Community Detection Graph

Visualizes detected communities using distinct colors.

### Interaction Strength Analysis

Edge colors represent interaction frequency:

* Blue → Low Interaction
* Yellow → Medium Interaction
* Red → High Interaction

### Influencer Visualization

Node size is proportional to PageRank score.

Large nodes indicate highly influential users.

---

## Machine Learning Pipeline

Dataset → Graph Construction → Node2Vec Embeddings → GCN Training → User Classification → Community Detection → Influencer Analysis

---

## Model Performance

The Graph Neural Network is evaluated using:

* Classification Accuracy
* Cross Entropy Loss

Example:

Accuracy = XX.XX%

(Replace with your actual model accuracy.)


## Future Enhancements

* Link prediction for friend recommendations
* Fraud and bot account detection
* Temporal graph analysis
* Graph Attention Networks (GAT)
* Recommendation systems
* Real-time social network monitoring

---

## Applications

* Social Media Analytics
* Community Discovery
* Influencer Marketing
* Recommendation Systems
* Fraud Detection
* User Behavior Analysis
