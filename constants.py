from transformers import BertTokenizer, BertForSequenceClassification, BertModel

# Set constants
epochs = 4
batch_size = 16

# Load tokenizer and model (using pre-trained bert IMDB model)
huggingface_model_name = "textattack/bert-base-uncased-imdb"
tokenizer = BertTokenizer.from_pretrained(huggingface_model_name)
pretrained_model = BertForSequenceClassification.from_pretrained(huggingface_model_name)
