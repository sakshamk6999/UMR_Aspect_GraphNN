from transformers import BertModel, BertTokenizer
import torch

def get_embeddings_with_pos(text):
  model_name = 'bert-base-uncased'

  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertModel.from_pretrained(model_name)

  embeddings = []

  num_tokens = []
  tokens = []

  # with torch.no_grad():

  for word in text.split():
    # print(word, end=' ')
    num_tokens.append(len(tokenizer.tokenize(word)))
    tokens.append(tokenizer.tokenize(word))
  # print('\n')

  # print("num words", len(text.split()), "num tokens", sum(num_tokens))
  tokenized = tokenizer(text, truncation=True, return_tensors="pt")
  # print("tokenized shape", tokenized['input_ids'].shape)
  embedded_input = model(**tokenized)
  # print("embedded shape", embedded_input.last_hidden_state.shape)

  curr = 1
  for i in range(len(num_tokens)):
    # print(tokens[i], num_tokens[i], curr, curr + num_tokens[i])
    embeddings.append(torch.sum(embedded_input.last_hidden_state[0][curr:curr+num_tokens[i]], dim=0))
    curr += num_tokens[i]

  embeddings = torch.stack(embeddings)
  # print("shape", embeddings.shape)
  return embeddings

def get_tokenized_with_pos(text):
  model_name = 'bert-base-uncased'

  tokenizer = BertTokenizer.from_pretrained(model_name)

  tokenized = tokenizer(text, truncation=True, return_tensors="pt", padding=True)

  return tokenized