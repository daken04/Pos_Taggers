from conllu import parse
from collections import Counter
import numpy as np
import re
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################################
################################################
## Preprocess Data

def load_and_preprocess_conllu(file_path):
    # Read the .conllu file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Parse the data
    parsed_data = parse(data)

    # Extract required information and lowercase words
    sentences = []
    word_freqs = Counter()
    for sentence in parsed_data:
        processed_sentence = []
        # processed_sentence = [('<start>', '<start>')]  # Start token
        # for _ in range(p-1):
        #   processed_sentence.append(('<start>', '<start>'))
        for token in sentence:
            word = token['form'].lower()
            pos_tag = token['upos']
            processed_sentence.append((word, pos_tag))
            word_freqs[word] += 1
        # processed_sentence.append(('<end>', '<end>'))  # End token
        # for _ in range(s-1):
        #   processed_sentence.append(('<end>', '<end>'))
        sentences.append(processed_sentence)

    # Replace infrequent words with OOV token
    sentences_with_oov = []
    for sentence in sentences:
        processed_sentence = []
        for word, pos_tag in sentence:
            if word_freqs[word] <= 3 and word not in ['<start>', '<end>']:
                processed_sentence.append(('<oov>', pos_tag))
            else:
                processed_sentence.append((word, pos_tag))
        sentences_with_oov.append(processed_sentence)

    return sentences_with_oov

file_path = 'en_atis-ud-train.conllu'
preprocessed_sentences = load_and_preprocess_conllu(file_path)
# print(preprocessed_sentences)  # Print the first preprocessed sentence

dev_file_path = 'en_atis-ud-dev.conllu'
test_file_path = 'en_atis-ud-test.conllu'

# Preprocess sentences for dev and test sets
preprocessed_sentences_dev = load_and_preprocess_conllu(dev_file_path)
preprocessed_sentences_test = load_and_preprocess_conllu(test_file_path)

# Placeholder for collecting all unique tags
all_tags = set()

for dataset in [preprocessed_sentences, preprocessed_sentences_dev, preprocessed_sentences_test]:
    for sentence in dataset:
        tags = {tag for _, tag in sentence if tag not in ['<start>', '<end>', '<oov>']}
        all_tags.update(tags)

# Now creating tag_to_ix including all tags
tag_to_ix = {tag: ix for ix, tag in enumerate(sorted(all_tags))}

def load_glove_embeddings(glove_file_path):
    embeddings_dict = {}
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict


glove_path = 'glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_path)

def build_embedding_map(preprocessed_sentences, glove_embeddings, embedding_dim=100):
    embedding_map = {}
    for sentence in preprocessed_sentences:
        for word, _ in sentence:
            if word not in embedding_map:  # Check to prevent duplicate work
                # Retrieve the embedding for the word, use a zero vector if the word is not found
                embedding_map[word] = glove_embeddings.get(word, np.zeros(embedding_dim))
    return embedding_map

embedding_dim = 100
word_to_embedding_map = build_embedding_map(preprocessed_sentences, glove_embeddings, embedding_dim)
word_to_embedding_map['<start>'] = np.zeros(embedding_dim)
word_to_embedding_map['<end>'] = np.zeros(embedding_dim)
# Add an OOV embedding to the embedding map if not already present
if '<oov>' not in word_to_embedding_map:
    word_to_embedding_map['<oov>'] = np.zeros(embedding_dim)  


###############################################################
###############################################################
## Model

class POSFFNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='ReLU', dropout_rate=0.3):
        super(POSFFNN, self).__init__()

        # Initialize layers using ModuleList to dynamically add layers based on the configuration
        self.layers = nn.ModuleList()

        # Input layer
        prev_dim = input_dim  # Keep track of the dimension of the previous layer

        # Add hidden layers dynamically
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))  # Linear layer
            # Add activation
            if activation == 'ReLU':
                self.layers.append(nn.ReLU())
            elif activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'LeakyReLU':
                self.layers.append(nn.LeakyReLU())
            # Add dropout after activation
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim  # Update prev_dim for the next layer

        # Output layer (no dropout before the output layer)
        self.layers.append(nn.Linear(prev_dim, output_dim))
        self.log_softmax = nn.LogSoftmax(dim=1)  # Use log-softmax for the output layer

    def forward(self, x):
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x)
        return self.log_softmax(x)

# Hyperparameters
embed_dim = 100  # Dimension of GloVe embeddings
hidden_dim = [128, 256]  # Size of the hidden layer
output_dim = len(tag_to_ix)
  # Number of unique POS tags

# Calculate the total input dimension
p = 2
s = 2
total_input_dim = (p + 1 + s) * embed_dim

# Instantiate the model
model = POSFFNN(total_input_dim, hidden_dim, output_dim)

# Load the model
model.load_state_dict(torch.load('model_ppnn.pt'))

####################################################################
####################################################################
## Presprocess and predict tags of input sentence

def preprocess_sentence(sentence, p=2, s=2):
    # Tokenize and lowercase the sentence
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = sentence.lower().split()

    # Add <start> and <end> tokens
    words = ['<start>']*p + words + ['<end>']*s
    return words

def sentence_to_embeddings(sentence_words, word_to_embedding_map, embedding_dim=100):
    embeddings = []
    for word in sentence_words:
        # Use the embedding for the word or a zero vector if the word is OOV
        embedding = word_to_embedding_map.get(word, word_to_embedding_map.get('<oov>', np.zeros(embedding_dim)))
        embeddings.append(embedding)
    return embeddings

def prepare_input_tensor(sentence_embeddings, p=2, s=2, embedding_dim=100):
    X = []
    for i in range(p, len(sentence_embeddings) - s):
        context_embeddings = sentence_embeddings[i-p:i+s+1]
        input_vector = np.hstack(context_embeddings).flatten()
        X.append(input_vector)
    return torch.tensor(X, dtype=torch.float)

def predict_tags(input_tensor, model, ix_to_tag):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_indices = torch.max(outputs, 1)
        predicted_tags = [ix_to_tag[ix] for ix in predicted_indices.numpy()]
    return predicted_tags

# Example sentence
test_sentence = input("Enter your text: ")

# Step 1: Preprocess the sentence
preprocessed_words = preprocess_sentence(test_sentence, p=2, s=2)

# Step 2: Convert sentence to embeddings
sentence_embeddings = sentence_to_embeddings(preprocessed_words, word_to_embedding_map, embedding_dim=100)

# Step 3: Prepare input tensor
input_tensor = prepare_input_tensor(sentence_embeddings, p=2, s=2, embedding_dim=100)

# Convert tag indices back to tag strings (inverse of tag_to_ix)
ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}

# Step 4: Predict POS tags
predicted_tags = predict_tags(input_tensor, model, ix_to_tag)

# Print the original sentence and its predicted tags
for word, tag in zip(preprocessed_words[2:-2], predicted_tags):  # Exclude <start> and <end> tokens
    print(f"{word}: {tag}")

#########################################
#########################################
## Evaluating this model on Eval and Test set
def prepare_dataset(preprocessed_sentences, word_to_embedding_map, tag_to_ix, p=2, s=2, embedding_dim=100):
    X = []
    Y = []

    for sentence in preprocessed_sentences:
        # Adding start and end tokens to the sentence
        extended_sentence = [('<start>','<start>')]*p + sentence + [('<end>','<end>')]*s

        for i in range(p, len(sentence)-s):
            context_words = extended_sentence[i-p:i+s+1]
            #
            context_embeddings = [word_to_embedding_map.get(word, word_to_embedding_map['<oov>']) for word, _ in context_words]
            # Flatten the list of context embeddings to a single input vector
            input_vector = np.hstack(context_embeddings).flatten()
            X.append(input_vector)

            # Get the target label for the current word
            _, tag = sentence[i]
            Y.append(tag_to_ix[tag])

    # Convert lists to tensors
    X_train = torch.tensor(X, dtype=torch.float)
    Y_train = torch.tensor(Y, dtype=torch.long)

    return X_train, Y_train

def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []
    with torch.no_grad():  # No need to track gradients
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')  # Use 'weighted' for imbalanced classes
    recall_macro = recall_score(all_targets, all_predictions, average='macro')
    recall_micro = recall_score(all_targets, all_predictions, average='micro')
    f1_micro = f1_score(all_targets, all_predictions, average='micro')
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    return accuracy, f1_macro,f1_micro,all_predictions,all_targets,recall_macro, recall_micro

batch_size = 32
X_dev, Y_dev = prepare_dataset(preprocessed_sentences_dev, word_to_embedding_map, tag_to_ix, p=2, s=2, embedding_dim=100)
X_test, Y_test = prepare_dataset(preprocessed_sentences_test, word_to_embedding_map, tag_to_ix, p=2, s=2, embedding_dim=100)
dev_dataset = TensorDataset(X_dev, Y_dev)
dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_dataset = TensorDataset(X_test, Y_test)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

entry=input("Do you want Eval Matrix like accuracy, recall, and F1 scores (micro, macro) (Y/n) : ")
if entry in ["Y","y"]:
    dev_accuracy, dev_f1_macro,dev_f1_micro, dev_y_pred, dev_y_true,dev_recall_macro, dev_recall_micro  = evaluate_model(model, dev_data_loader)
    print(f"Dev Set Evaluation - Accuracy: {dev_accuracy:.4f}, F1 Score (Macro): {dev_f1_macro:.4f}, F1 Score (Micro): {dev_f1_micro:.4f}, Recall (Macro) : {dev_recall_macro:.4f}, Recall (Micro): {dev_recall_macro:.4f}")
    test_accuracy, test_f1_macro,test_f1_micro, test_y_pred, test_y_true,test_recall_macro, test_recall_micro  = evaluate_model(model, test_data_loader)
    print(f"Test Set Evaluation - Accuracy: {test_accuracy:.4f}, F1 Score (Macro): {test_f1_macro:.4f}, F1 Score (Micro): {test_f1_micro:.4f} , Recall (Macro) : {test_recall_macro:.4f}, Recall (Micro): {test_recall_macro:.4f}")

    #code to plot confusion_matrix for dev
    cm = confusion_matrix(dev_y_true,dev_y_pred)
    ConfusionMatrixDisplay(cm).plot()

    #code to plot confusion_matrix for test
    cm = confusion_matrix(test_y_true,test_y_pred)
    ConfusionMatrixDisplay(cm).plot()


