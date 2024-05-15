import os
import torch
import time

from BlockImportance.WeightedSumAttentionApproach import get_approach_1_model
from BlockImportance.BlockOutputAttentionApproach import get_approach_2_model
from helpers import report_most_important_blocks, save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weighted_summation(approach_1_attention_weights, approach_2_attention_weights):
    # Weights for each approach
    w1 = 0.5  # Weight for approach_1
    w2 = 0.5  # Weight for approach_2

    # Compute weighted average
    weighted_average = w1 * approach_1_attention_weights + w2 * approach_2_attention_weights

    # Sorting the blocks by importance
    sorted_indices = torch.argsort(weighted_average, descending=True)
    sorted_weighted_average = weighted_average[sorted_indices]

    # Print the weighted average importance of encoder blocks in a ranked format
    print("Ranked importance of encoder blocks:")
    for idx, value in enumerate(sorted_weighted_average):
        block_number = sorted_indices[idx].item() + 1  # Adding 1 because block numbers are 1-indexed
        print(f"Block {block_number}: {value.item():.4f}")

    return sorted_indices

def find_most_important_blocks(pretrained_model, train_dataloader, test_dataloader, use_saved_models=False):
    block_scores = []
    if use_saved_models:
        block_scores = report_from_saved_models()
    else:
        block_scores = train_to_find(pretrained_model, train_dataloader, test_dataloader)

    if block_scores == []:
        print("Error: No block scores found.")
        return []

    # Return weighted summation of block_scores
    return weighted_summation(block_scores[0], block_scores[1])

def report_from_saved_models():
    current_script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(current_script_dir, 'saved_models')
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: The directory {base_dir} does not exist.")
        return []
    # Get a list of directories in the base directory, safely
    try:
        directories = next(os.walk(base_dir))[1]
    except StopIteration:
        print(f"No directories found in {base_dir}.")
        return []

    # Sort directories
    directories.sort()
    if directories:
        # Get the latest directory
        latest_dir = directories[-1]
        latest_dir_path = os.path.join(base_dir, latest_dir)
    else:
        print("No directories found after sorting.")
        return []
    
    # Load BertWithWeightedSumAttention from approach_1_model.pt
    approach_1_model = torch.load(os.path.join(latest_dir_path, 'approach_1_model.pt'), map_location=device)
    approach_1_model.eval()
    attention_weights_1 = approach_1_model.block_attention.weights
    softmax_attention_weights_1 = report_most_important_blocks(attention_weights_1)
    print("Loaded approach 1 model.")
    # Load BertWithAttentionAfterEachEncoder from approach_2_model.pt
    approach_2_model = torch.load(os.path.join(latest_dir_path, 'approach_2_model.pt'), map_location=device)
    approach_2_model.eval()
    attention_weights_2 = approach_2_model.scaling_attention.weights    
    softmax_attention_weights_2 = report_most_important_blocks(attention_weights_2)
    print("Loaded approach 2 model.")
       
    return [softmax_attention_weights_1, softmax_attention_weights_2]
    

def train_to_find(pretrained_model, train_dataloader, test_dataloader):
    save_path = time.strftime("%Y%m%d-%H%M%S")

    # Approach 1: Injecting attention over the blocks of the BERT model
    print("Training approach 1 model...")
    approach_1_model, attention_weights_1 = get_approach_1_model(pretrained_model, train_dataloader)
    print("Most important blocks by sorting block weights from high-to-low:")
    softmax_attention_weights_1 = report_most_important_blocks(attention_weights_1)
    print("Saving...")
    save_model(approach_1_model, save_path, "approach_1_model")

    # Approach 2: Add attention between encoder blocks before passing to the next block
    print("Training approach 2 model...")
    approach_2_model, attention_weights_2 = get_approach_2_model(pretrained_model, train_dataloader)
    print("Most important blocks by sorting block weights from high-to-low:")
    softmax_attention_weights_2 = report_most_important_blocks(attention_weights_2)    
    print("Saving...")
    save_model(approach_2_model, save_path, "approach_2_model")

    return [softmax_attention_weights_1, softmax_attention_weights_2]