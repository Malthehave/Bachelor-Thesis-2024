import torch

from helpers import convert_to_dataloader, count_number_of_parameters, slice_dataset, get_dataset, evaluate_model, stratified_split
from constants import pretrained_model, tokenizer
from find_most_important_blocks import find_most_important_blocks
from optimized_model import get_optimized_model
from baseline_pruning import prune_and_evaluate

torch.manual_seed(17)

def main():
    # Get dataset
    train_dataset, test_dataset = get_dataset(tokenizer)

    should_slice_dataset = True
    if should_slice_dataset:
        # Cut dataset
        train_val_dataset = slice_dataset(train_dataset, 10)
        test_dataset = slice_dataset(test_dataset, 5)
        # Split the test dataset into test and val
        train_dataset, val_dataset = stratified_split(train_val_dataset, test_size=0.5)   
        
    # Convert to dataloader
    train_dataloader = convert_to_dataloader(train_dataset)
    test_dataloader = convert_to_dataloader(test_dataset)
    val_dataloader = convert_to_dataloader(val_dataset)

    # Baseline pruning
    prune_and_evaluate(pretrained_model, val_dataloader)    

    # Find most important blocks
    block_scores = find_most_important_blocks(pretrained_model, train_dataloader, test_dataloader, use_saved_models=False)
    print("Number of parameters in full model including embeddings:", count_number_of_parameters(pretrained_model))
    print("Number of parameters in one encoder block of full model:", count_number_of_parameters(pretrained_model.bert.encoder.layer[0]))    

    print("Most important blocks: ", block_scores)

    # Build optimized model based on most important blocks
    optimized_model, parameter_performance_tradeoff = get_optimized_model(pretrained_model, train_dataloader, test_dataloader, val_dataloader, block_scores)
    print("Parameter performance tradeoff:")
    for block_performance in parameter_performance_tradeoff:
        print("Block:", block_performance["block_idx"])
        print("Reduction in parameters:", block_performance["reduction"])
        print("Accuracy:", block_performance["accuracy"])
        print("Normalized block importance scores:", block_performance["normalized_block_importance_scores"])
    
    print("Percentage reduction in parameters:", 1 - (count_number_of_parameters(optimized_model) / count_number_of_parameters(pretrained_model)))
    print("Number of parameters in optimized model including embeddings:", count_number_of_parameters(optimized_model))    

    # Evaluate original model
    acc = evaluate_model(pretrained_model, test_dataloader)
    print("Accuracy of original model:", acc)

    return optimized_model, block_scores

if __name__ == "__main__":
    optimized_model, block_scores = main()
