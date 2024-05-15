from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from constants import batch_size

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_text(model, tokenizer, text):
    # Move model to the correct device
    model = model.to(device)
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    logits = model(input_ids, attention_mask)
    return logits

def count_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataset(tokenizer, dataset_name="imdb", cache=True):
    current_script_dir = os.path.dirname(__file__)    
    train_data_file_path = os.path.join(current_script_dir, 'saved_datasets/tokenized_datasets', f'{dataset_name}_train')    
    val_data_file_path = os.path.join(current_script_dir, 'saved_datasets/tokenized_datasets', f'{dataset_name}_val')    
    if cache:
        # Check if the dataset is already saved
        if os.path.exists(train_data_file_path) and os.path.exists(val_data_file_path):
            print("Using cached dataset")            
            train = load_from_disk(train_data_file_path)
            val = load_from_disk(val_data_file_path)            
            return train, val
            
    dataset = load_dataset(dataset_name)
    # Tokenize the input texts
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if cache:
        # Save the tokenized datasets
        save_dataset(tokenized_datasets['train'], 'tokenized_datasets', f'{dataset_name}_train')
        save_dataset(tokenized_datasets['test'], 'tokenized_datasets', f'{dataset_name}_val')

    return tokenized_datasets['train'], tokenized_datasets['test']

def slice_dataset(dataset, max_rows):
    """
    Slices the given dataset to only include max_rows number of rows. We balance the dataset. I.e. taking the first max_rows/ rows from the first part of the dataset and the last max_rows/2 rows from the last part of the dataset.
    
    Args:
    dataset (Dataset): The dataset to be sliced. Assumed to be a PyTorch Dataset or similar.
    max_rows (int): Maximum number of rows to keep in the sliced dataset.
    
    Returns:
    Dataset: A new dataset containing at most max_rows entries from the original dataset.
    """
    # Ensure the max_rows does not exceed the actual size of the dataset
    max_rows = min(max_rows, len(dataset))

    # Calculate the indices to keep
    indices = list(range(max_rows//2)) + list(range(len(dataset) - max_rows//2, len(dataset)))

    # Create a subset of the dataset
    sliced_dataset = Subset(dataset, indices)

    
    return sliced_dataset

def stratified_split(dataset, test_size=0.5):
    """
    Splits the dataset into two stratified subsets based on the class labels.
    
    Args:
    dataset (Dataset): The dataset to be split. Assumed to be a PyTorch Dataset or similar.
    test_size (float): The proportion of the dataset to include in the test split.
    
    Returns:
    (Dataset, Dataset): Two datasets, the first being the smaller split and the second the larger.
    """

    # Extract labels
    labels = [x['label'] for x in dataset]

    # Generate indices for splitting
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=17,
        stratify=labels
    )

    # Create subsets for train and test
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    return train_subset, test_subset

def convert_to_dataloader(dataset, batch_size=batch_size, shuffle=True):    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def report_most_important_blocks(values, descending=True):
    # Run softmax over the values
    values = torch.nn.functional.softmax(values, dim=0)
    print(f"Block importance ranked by sorting with descending={descending}:")
    sorted_values, sorted_indices = torch.sort(torch.tensor(values), descending=descending)
    for i, idx in enumerate(sorted_indices):
        print(f"#{i+1}. Block {idx+1}, Value: {sorted_values[i].item()}")
    return values


def build_dataset(pretrained_model, train_dataloader, dataset_name, cache=True):
    # Check if the dataset is already saved
    if cache:
        current_script_dir = os.path.dirname(__file__)    
        train_block_inputs_file_path = os.path.join(current_script_dir, f'saved_datasets/{dataset_name}', 'inputs')    
        train_block_outputs_file_path = os.path.join(current_script_dir, f'saved_datasets/{dataset_name}', 'outputs')    
        if os.path.exists(train_block_inputs_file_path) and os.path.exists(train_block_outputs_file_path):
            print("Using cached dataset")            
            train_block_inputs = load_from_disk(train_block_inputs_file_path)
            train_block_outputs = load_from_disk(train_block_outputs_file_path)            
            return train_block_inputs, train_block_outputs
    pretrained_model.eval()  # Ensure the model is in evaluation mode
    all_batch_inputs = []
    all_batch_outputs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, data in tqdm(enumerate(train_dataloader), desc="Building Dataset"):
        input_ids = pad_sequence([LongTensor(i) for i in data['input_ids']]).to(device)
        attention_mask = pad_sequence([LongTensor(i) for i in data['attention_mask']]).to(device)
        
        with torch.no_grad():
            # Use automatic mixed precision
            # with autocast():
            output = pretrained_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        
        hidden_states = output.hidden_states  # Tuple of length 13 (embedding output + 12 layers)
        block_outputs = torch.stack(hidden_states[1:], dim=1).cpu()  # Move to CPU after stacking
        block_inputs = torch.stack(hidden_states[:-1], dim=1).cpu()  # Move to CPU after stacking

        all_batch_inputs.append(block_inputs)
        all_batch_outputs.append(block_outputs)

    # Concatenate all the batches to form the final dataset (on CPU)
    input_dataset = torch.cat(all_batch_inputs, dim=0)
    del all_batch_inputs
    if cache:
        save_dataset(input_dataset, dataset_name, "inputs")

    output_dataset = torch.cat(all_batch_outputs, dim=0)
    del all_batch_outputs
    if cache:
        save_dataset(output_dataset, dataset_name, "outputs")

    return input_dataset, output_dataset

def save_model(model, path, name):
    current_script_dir = os.path.dirname(__file__)    
    model_path = os.path.join(current_script_dir, f"saved_models/{path}", f"{name}.pt")    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)        
    torch.save(model, model_path)

def evaluate_model(model, test_dataloader, debug=False):
    model = model.to(device)
    model = model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            input_ids = pad_sequence([LongTensor(i) for i in data['input_ids']]).to(device)
            attention_mask = pad_sequence([LongTensor(i) for i in data['attention_mask']]).to(device)
            labels = torch.tensor(data['label']).to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if debug:
                print("outputs", outputs)
                print("predicted", predicted)
                print("labels", labels)
    return correct / total


def save_dataset(dataset, path, name):
    current_script_dir = os.path.dirname(__file__)    
    dataset_path = os.path.join(current_script_dir, f"saved_datasets/{path}", f"{name}")    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)    
    dataset.save_to_disk(dataset_path)