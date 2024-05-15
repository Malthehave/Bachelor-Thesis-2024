import torch
import copy
import pandas as pd
from tqdm import tqdm

from helpers import evaluate_model, count_number_of_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_zero_parameters(model):
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    return zero_params

def get_pruning_statistics(model):
    total_params = count_number_of_parameters(model)
    zero_params = count_zero_parameters(model)
    pruned_percentage = (zero_params / total_params) * 100    
    return {
        'Total Parameters': total_params,
        'Pruned Parameters': zero_params,
        'Percentage Pruned': f"{pruned_percentage:.2f}%",        
    }   

def efficient_quantile(parameters, sparsity, num_splits=10):
    # Split parameters into smaller chunks
    split_tensors = torch.split(parameters, split_size_or_sections=int(len(parameters) / num_splits))
    
    quantiles = []
    for tensor in split_tensors:
        quantiles.append(torch.quantile(tensor.abs(), sparsity))
    
    # Return the median of the quantiles as an approximation
    return torch.median(torch.tensor(quantiles))

def prune_by_magnitude(model, sparsity=0.2):
    # Copy model
    pruned_model = copy.deepcopy(model).to(device)
    # Step 1: Gather all parameters
    all_parameters = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

    # Multiply sparsity by 1.053962901 to account for the fact that the distribution of weights is not uniform
    sparsity *= 1.053962901

    # Step 2: Determine the threshold for the desired sparsity
    threshold = efficient_quantile(all_parameters, sparsity)
    
    # Step 3: Prune the model
    pruned_model = prune_by_threshold(pruned_model, threshold)

    return pruned_model


def prune_by_threshold(model, threshold=0.01):
    # Copy model
    pruned_model = copy.deepcopy(model).to(device)

    # Prune the weights in each layer of the model
    for param in pruned_model.parameters():
        if param.requires_grad:
            mask = param.abs() > threshold
            param.data *= mask.float()

    return pruned_model

def prune_randomly(model, sparsity=0.2):
    # Copy model to device
    pruned_model = copy.deepcopy(model).to(device)

    # Randomly generate masks for each parameter
    for param in pruned_model.parameters():
        if param.requires_grad:
            mask = torch.rand_like(param.data) > sparsity  # Keep (1-sparsity)% of the weights
            param.data *= mask.float()

    return pruned_model

def prune_and_evaluate(model, val_dataloader):
    sparsity_levels = [0.0593, 0.1186, 0.1779, 0.2373, 0.2969, 0.3559, 0.4152, 0.4745, 0.5338, 0.5931, 0.6524, 0.7118]    
    results = []


    for level in tqdm(sparsity_levels, desc="Pruning Levels"):
        # Prune the model by magnitude
        pruned_model = prune_by_magnitude(model, sparsity=level)
        mag_pruning_stats = get_pruning_statistics(pruned_model)
        acc_magnitude = evaluate_model(pruned_model, val_dataloader)
        results.append({'Sparsity Level': level, 'Pruning Type': 'Magnitude', **mag_pruning_stats, 'Accuracy': acc_magnitude})
        del pruned_model

        # Prune the model randomly
        pruned_model = prune_randomly(model, sparsity=level)
        rand_pruning_stats = get_pruning_statistics(pruned_model)
        acc_random = evaluate_model(pruned_model, val_dataloader)
        results.append({'Sparsity Level': level, 'Pruning Type': 'Random', **rand_pruning_stats, 'Accuracy': acc_random})
        del pruned_model

    # Display results in a DataFrame for better readability
    results_df = pd.DataFrame(results)
    print(results_df)

