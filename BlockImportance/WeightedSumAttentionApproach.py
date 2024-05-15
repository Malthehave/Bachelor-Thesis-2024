import torch
from torch import nn
from torch.nn import functional as F
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from constants import epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedSumBlockAttention(nn.Module):
    '''
    Injects a learnable attention mechanism over the blocks of a sequence.
    The output is a weighted summation of all block outputs.
    NOTE: This is not the same as applying attention to the output of each encoder block before feeding them to the next block. That is "WEIGHTED OUTPUT" approach as seen in BlockOutputAttentionApproach.py
    '''
    def __init__(self, num_blocks):
        super(WeightedSumBlockAttention, self).__init__()
        # The weights are initialized uniformly as 1/num_blocks
        self.weights = nn.Parameter(torch.ones(num_blocks) / num_blocks)

    def forward(self, block_outputs):
        # block_outputs is a list of tensors, each of shape [batch_size, seq_len, hidden_dim] 
        weighted_sum = torch.stack([self.weights[i] * block_outputs[i] for i in range(len(self.weights))], dim=-1).sum(dim=-1)
        return weighted_sum
    

class BertWithWeightedSumAttention(nn.Module):
    def __init__(self, bert_model):
        super(BertWithWeightedSumAttention, self).__init__()
        self.bert = bert_model.bert
        self.dropout = bert_model.dropout
        self.classifier = bert_model.classifier

        self.block_attention = WeightedSumBlockAttention(num_blocks=bert_model.config.num_hidden_layers)

    def forward(self, input_ids, attention_mask=None):
        # Request hidden states from the bert model
        bert_output = self.bert(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            output_hidden_states=True)

        # Now, hidden_states will be a tuple containing the output of each layer, including the embedding layer
        hidden_states = bert_output.hidden_states
        # Iterate over the hidden states. Note that hidden_states[0] is the embedding layer output, we start from hidden_states[1:]
        block_outputs = [layer_output for layer_output in hidden_states[1:]]

        weighted_sum = self.block_attention(block_outputs) # Dimension of weighted_sum is [batch_size, seq_len, hidden_dim] 
        pooled_output = self.bert.pooler(weighted_sum)
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
        # for i, batch in enumerate(train_dataloader):
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

def get_approach_1_model(pretrained_model, train_dataloader):
    model = BertWithWeightedSumAttention(bert_model=pretrained_model).to(device)
    trained_model = train_model(model, train_dataloader, epochs=epochs)
    attention_weights = trained_model.block_attention.weights    
    return trained_model, attention_weights

