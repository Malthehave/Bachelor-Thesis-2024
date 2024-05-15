import torch
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
from tqdm import tqdm
import copy

from helpers import convert_to_dataloader, evaluate_model, count_number_of_parameters, save_model
from constants import epochs
from BlockImportance.BlockSensitivity import compute_sensitivity

torch.manual_seed(17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub_network_bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
config = BertConfig.from_pretrained("prajjwal1/bert-tiny") # We will use the same encoder architecture as prajjwal1/bert-tiny (2 layers, 128 hidden size, 2 attention heads)
sub_network_bert = BertModel(config)

# Define subnetwork structure
class SubNetworkWithProjection(torch.nn.Module):
    def __init__(self, sub_network, original_block_hidden_size):
        super(SubNetworkWithProjection, self).__init__()
        self.input_linear = torch.nn.Linear(original_block_hidden_size, sub_network.config.hidden_size)
        # Clone the sub_network
        sub_network = copy.deepcopy(sub_network)
        # Delete the embedding and pooler from the sub_network
        del sub_network.embeddings
        del sub_network.pooler
        self.sub_network = sub_network
        self.output_linear = torch.nn.Linear(sub_network.config.hidden_size, original_block_hidden_size)

    # def forward(self, x):
    def forward(self, x, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, output_hidden_states=False):
        hidden_state = x
        batch_size, seq_length, feature_size = x.shape

        x = x.reshape(-1, feature_size) # Flatten the batch and sequence length dimensions        

        x = self.input_linear(x) # Now x should be [batch_size*seq_length, 128]

        x = x.reshape(batch_size, seq_length, -1) # Reshape back for sub_network if needed        
        
        # pass x to sub_network encoder only
        outputs = self.sub_network.encoder(x, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_value, output_attentions=output_attentions, output_hidden_states=output_hidden_states)        

        # Use outputs.last_hidden_state for subsequent operations
        x = outputs.last_hidden_state # Shape: [batch_size, seq_length, hidden_size]        
        
        x = x.reshape(-1, x.size(-1))
        x = self.output_linear(x) # Project back to original feature size        

        x = x.reshape(batch_size, seq_length, -1)
        x.unsqueeze_(1) # Add a dimension for the number of layers

        # Add residual connection
        x = x + hidden_state

        return x

def get_subnetwork(pretrained_model):
    model = SubNetworkWithProjection(sub_network_bert, pretrained_model.config.hidden_size)
    model = model.to(device)
    model = model.train()
    return model

def substitute_and_train(model, train_dataloader, block_index_to_substitute):   
    sub_network = get_subnetwork(model)

    # Substitute the block
    model.bert.encoder.layer[block_index_to_substitute] = sub_network

    # Separate parameters
    sub_network_params = model.bert.encoder.layer[block_index_to_substitute].parameters()
    base_params = [p for n, p in model.named_parameters() if not n.startswith(f'bert.encoder.layer.{block_index_to_substitute}')]


    # Create parameter groups with different learning rates
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 2e-5},  # Lower learning rate for the pre-trained parts
        {'params': sub_network_params, 'lr': 1e-4}  # Higher learning rate for the newly added sub-network
    ])

    # Fine tune the modified model    
    model = model.to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True, position=1):
        for i, data in tqdm(enumerate(train_dataloader), desc="Batches", leave=False, position=0):
            input_ids = pad_sequence([LongTensor(i) for i in data['input_ids']]).to(device)
            attention_mask = pad_sequence([LongTensor(i) for i in data['attention_mask']]).to(device)
            labels = LongTensor(data['label']).to(device)
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model

def get_optimized_model(pretrained_model, train_dataloader, test_dataloader, val_dataloader, sorted_block_importance_indices):
    # Copy the original model
    model = copy.deepcopy(pretrained_model).to(device)
    
    # Reverse the block importance so we train the least important blocks first
    reversed_block_importance_indices = torch.flip(sorted_block_importance_indices, dims=[0])

    # Save a dict of parameter reduction and test accuracy for each
    parameter_performance_tradeoff = []

    # Substitute every block in reversed_block_importance_indices
    for block_idx in reversed_block_importance_indices:
        print("Substituting block:", int(block_idx + 1))
        model = substitute_and_train(model, train_dataloader, block_idx)        
        param_reduction = 1 - (count_number_of_parameters(model) / count_number_of_parameters(pretrained_model))
        print("Reduction in parameters:", param_reduction)
        # Evaluate the network accuracy
        acc = evaluate_model(model, test_dataloader, debug=False)
        print("Accuracy:", acc)

        # Run block sensitivity analysis on optimized model 
        normalized_importance_scores = compute_sensitivity(model, test_dataloader)

        parameter_performance_tradeoff.append({
            "block_idx": block_idx,
            "reduction": param_reduction,
            "accuracy": acc,
            "normalized_block_importance_scores": normalized_importance_scores
        })
        # Save the model to disk
        save_model(model, "optimized_models/run1/", block_idx)


    return model, parameter_performance_tradeoff