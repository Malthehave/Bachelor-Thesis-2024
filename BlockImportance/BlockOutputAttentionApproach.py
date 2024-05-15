import torch
from torch import nn
from torch.nn import functional as F
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from constants import epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlockOutputAttention(nn.Module):
    def __init__(self, num_blocks):
        '''
        Adds a learnable attention mechanism over the output of each encoder block.
        The input of a block is the scaled output of the previous block.
        The embeddings are not scaled.
        '''
        super(BlockOutputAttention, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_blocks) / num_blocks)

    def forward(self, block_output, layer_index):        
        score = self.weights
        # Multiply the output by the scaling factor for the current layer
        scaled_output = block_output * score[layer_index]
        return scaled_output

class BertWithAttentionAfterEachEncoder(nn.Module):
    def __init__(self, config, bert_model):
        super(BertWithAttentionAfterEachEncoder, self).__init__()

        self.config = config
        self.bert = bert_model.bert
        self.dropout = bert_model.dropout    
        self.classifier = bert_model.classifier    
        self.scaling_attention = BlockOutputAttention(config.num_hidden_layers)

    def forward(self, input_ids, attention_mask=None):
        # Standard BERT inputs
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)
        
        # We need to manually loop through each encoder layer
        hidden_states = self.bert.embeddings(input_ids) # hidden states is dimension 
        for i, layer_module in enumerate(self.bert.encoder.layer):            
            layer_outputs = layer_module(hidden_states, extended_attention_mask) # Tuple of (layer_output, token_output, attention_output)
            hidden_states = layer_outputs[0] # Get layer output of dimension (token_output, attention_output)
            # Apply the scaling attention here
            if i < self.config.num_hidden_layers:
                hidden_states = self.scaling_attention(hidden_states, i)

        # Apply the pooler
        pooled_output = self.bert.pooler(hidden_states)
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits
    

def train_model(model, train_dataloader, epochs=epochs):    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, batch in tqdm(enumerate(train_dataloader), desc="Batches", leave=False):
            # input_ids = batch['input_ids']
            input_ids = pad_sequence([LongTensor(i) for i in batch['input_ids']]).to(device)
            attention_mask = pad_sequence([LongTensor(i) for i in batch['attention_mask']]).to(device)
            # attention_mask = torch.tensor(batch['attention_mask']).to(device)
            labels = batch['label'].to(device)        
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

def get_approach_2_model(pretrained_model, train_dataloader):
    model = BertWithAttentionAfterEachEncoder(config=pretrained_model.config, bert_model=pretrained_model).to(device)
    trained_model = train_model(model, train_dataloader, epochs=epochs)
    attention_weights = trained_model.scaling_attention.weights    
    return trained_model, attention_weights