import torch
from torch import nn
from torch.nn import functional as F
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gradients_and_outputs_of_blocks(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    
    block_importance_scores = torch.zeros(model.config.num_hidden_layers)

    # Loop over the batches
    for batch in tqdm(dataloader, desc="Batches", leave=False):
        input_ids = pad_sequence([LongTensor(i) for i in batch['input_ids']]).to(device)
        attention_mask = pad_sequence([LongTensor(i) for i in batch['attention_mask']]).to(device)
        labels = batch['label'].to(device)

        with torch.set_grad_enabled(True):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            logits = outputs.logits
            loss = criterion(logits, labels)
            # Ensure gradients are retained for intermediate activations
            for layer_output in outputs.hidden_states[1:]:  # Exclude embeddings layer
                layer_output.retain_grad()

            loss.backward()

            # Compute importance scores for each block
            for i, block_output in enumerate(outputs.hidden_states[1:], 1):
                if block_output.grad is not None:                    
                    importance = torch.abs(block_output.grad * block_output).sum()                    
                    block_importance_scores[i - 1] += importance.item()


    return block_importance_scores

def normalize_importance_scores(importance_scores):
    return importance_scores / importance_scores.sum()

def compute_sensitivity(optimized_model, dataloader):
    model_copy = copy.deepcopy(optimized_model)
    importance_scores = get_gradients_and_outputs_of_blocks(model_copy, dataloader)
    normalized_importance_scores = normalize_importance_scores(importance_scores)
    plot_heatmap(normalized_importance_scores)
    return normalized_importance_scores


def plot_heatmap(normalized_importance_scores):
    sns.set_theme()
    plt.figure(figsize=(10, 5))
    sns.heatmap(normalized_importance_scores.unsqueeze(0), annot=True, fmt=".2f", cmap="viridis", cbar=False)
    plt.xlabel("Block number")
    plt.ylabel("Importance")
    plt.title("Normalized importance scores of encoder blocks")
    plt.show()


def combined_heatmap(list_of_normalized_importance_scores):
    sns.set_theme()
    plt.figure(figsize=(10, 5))
    for i, normalized_importance_scores in enumerate(list_of_normalized_importance_scores):
        sns.heatmap(normalized_importance_scores.unsqueeze(0), annot=True, fmt=".2f", cmap="viridis", cbar=False)
        plt.xlabel("Block number")
        plt.ylabel("Importance")
        plt.title(f"Normalized importance scores of encoder blocks for model {i}")
        plt.show()