# A-Deep-Learning-Approach-for-Hyperparameterless-Graph-Coloring

This repository contains code to implement and train PI-SAGE model for selecting q color through two algorithms MAXC-q1 and MAXC-q2 as well as analysis of learning rate. The GNN models supported are GraphSAGE and GraphConv, designed using PyTorch and DGL (Deep Graph Library).

#Prerequisites:
Required Libraries:
Python 3.8 or above
torch==2.3.0
dgl (compatible with the installed PyTorch version)
Other Python libraries: numpy, networkx, os, warnings, etc.

#Hardware: T4-GPU and colab

#Usage:
1. Graph Construction: Parse the graph definition from .col files.
2. Model Initialization: Create and configure the GraphSAGE or GraphConv model.
3. Training: Train the GNN using the provided hyperparameters and stop early if conditions are met.
4. Evaluation: Compute the graph coloring cost and adjacency matrix.

#Key Configurations

Input Graph File: Change the problem_file variable to load a different .col file.
Hyperparameters:
Adjust hypers for GNN-specific settings (e.g., model, dim_embedding, dropout). We have changed and use different learning rate to analyze them
Modify solver_hypers for training settings (e.g., tolerance, number_epochs, patience).
Device Selection: The script automatically detects and uses a GPU if available. Update TORCH_DEVICE if needed.
