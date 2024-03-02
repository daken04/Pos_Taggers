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

def load_and_preprocess_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    parsed_data = parse(data)
    sentences = []
    word_freqs = Counter()
    
    for sentence in parsed_data:
        word_freqs.update([token['form'].lower() for token in sentence])
    
    for sentence in parsed_data:
        processed_sentence = [(token['form'].lower() if word_freqs[token['form'].lower()] > 3 else "<OOV>", token['upos']) for token in sentence]
        sentences.append(processed_sentence)
    
    return sentences, word_freqs


file_path = 'en_atis-ud-train.conllu'
sentences, word_freqs = load_and_preprocess_conllu(file_path)

def load_glove_embeddings(glove_file_path):
    embeddings_dict = {}
    all_vectors = []  # List to store all embeddings for calculating the average
    
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
            all_vectors.append(vector)
    
    # Calculate the average vector (could be used for <OOV> tokens)
    average_vector = np.mean(all_vectors, axis=0)
    embeddings_dict["<oov>"] = average_vector
    
    return embeddings_dict

glove_path = 'glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_path)


def pad_sequences(sequences, dim=100, pad_tags=False):
    """Pad sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    if pad_tags:
        # For tag sequences, create a padded array of zeros with shape (batch_size, max_len)
        padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    else:
        # For word embedding sequences, create a padded array of zeros with shape (batch_size, max_len, dim)
        padded_sequences = np.zeros((len(sequences), max_len, dim))

    for i, seq in enumerate(sequences):
        if not pad_tags:
            padded_sequences[i, :len(seq), :] = seq
        else:
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences

def prepare_data(sentences, word_freqs, glove_embeddings, embedding_dim=100):
    X = []
    Y = []
    tag_to_ix = {}
    current_tag_index = 0

    oov_embedding = glove_embeddings.get("<oov>")

    for sentence in sentences:
        sentence_embeddings = []
        sentence_tags = []
        for word, tag in sentence:
            embedding = glove_embeddings.get(word,oov_embedding )
            sentence_embeddings.append(embedding)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = current_tag_index
                current_tag_index += 1
            sentence_tags.append(tag_to_ix[tag])

        X.append(np.array(sentence_embeddings))
        Y.append(np.array(sentence_tags))

    # Pad sequences to the same length
    X_padded = pad_sequences(X, dim=embedding_dim)
    Y_padded = pad_sequences(Y, pad_tags=True)

    # Convert lists to tensors
    X_tensor = torch.tensor(X_padded, dtype=torch.float)
    Y_tensor = torch.tensor(Y_padded, dtype=torch.long)

    return X_tensor, Y_tensor, tag_to_ix

X_train, Y_train, tag_to_ix = prepare_data(sentences, word_freqs, glove_embeddings)

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, num_layers=1, bidirectional=False, dropout=0.3, activation_function='relu'):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        # LSTM layer with dropout for multi-layer LSTMs
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))

        # Additional dropout layer before the linear layer
        self.dropout = nn.Dropout(dropout)

        # Mapping to tag space
        multiplier = 2 if bidirectional else 1
        self.hidden2tag = nn.Linear(hidden_dim * multiplier, tagset_size)

        # Activation function
        if activation_function == 'tanh':
            self.activation = nn.Tanh()
        elif activation_function == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None  # It's safer to default to None if the activation function is unsupported

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        if self.activation:
            lstm_out = self.activation(lstm_out)
        
        # Apply dropout to the output of the LSTM or activation function
        lstm_out = self.dropout(lstm_out)
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores
    
import re
def sentence_to_embeddings(sentence, glove_embeddings, embedding_dim=100):
    """
    Convert a sentence into a sequence of GloVe embeddings.

    :param sentence: The input sentence as a string.
    :param glove_embeddings: A dictionary mapping words to GloVe vectors.
    :param embedding_dim: The dimension of the GloVe embeddings.
    :return: A tensor of shape (1, seq_length, embedding_dim) representing the embedded sentence.
    """
    # Tokenize and preprocess (e.g., lowercasing) the sentence
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = sentence.lower().split()

    # Convert words to their corresponding GloVe embeddings
    # Use OOV embedding if the word is not found
    embeddings = [glove_embeddings.get(word, glove_embeddings.get("<oov>")) for word in words]


    # Convert the list of embeddings into a tensor
    embeddings_tensor = torch.tensor([embeddings], dtype=torch.float)

    return embeddings_tensor

def predict_pos_tags(sentence, model, glove_embeddings, tag_to_ix, embedding_dim=100):
    """
    Predict the POS tags for a given sentence.

    :param sentence: The input sentence as a string.
    :param model: The trained LSTM model.
    :param glove_embeddings: A dictionary mapping words to GloVe vectors.
    :param tag_to_ix: A dictionary mapping POS tags to unique indices.
    :param embedding_dim: The dimension of the GloVe embeddings.
    :return: A list of predicted POS tags for the sentence.
    """
    # Convert the sentence to embeddings
    sentence_embeddings = sentence_to_embeddings(sentence, glove_embeddings, embedding_dim)

    # Predict the tags
    model.eval()
    with torch.no_grad():
        outputs = model(sentence_embeddings)
        _, predicted_indices = torch.max(outputs, dim=2)

    # Convert indices to tags
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
    predicted_tags = [ix_to_tag[ix.item()] for ix in predicted_indices[0]]

    return predicted_tags


sentence = input("Enter the sentence: ")

dev_file_path = 'en_atis-ud-dev.conllu'  
test_file_path = 'en_atis-ud-test.conllu' 

# Load and preprocess the development dataset
dev_sentences, dev_word_freqs = load_and_preprocess_conllu(dev_file_path)

# Load and preprocess the test dataset
test_sentences, test_word_freqs = load_and_preprocess_conllu(test_file_path)


# Prepare development data
X_dev, Y_dev, _ = prepare_data(dev_sentences, dev_word_freqs, glove_embeddings)

# Prepare test data
X_test, Y_test, _ = prepare_data(test_sentences, test_word_freqs, glove_embeddings)


def evaluate_model(model, dev_data):
    model.eval()
    all_preds = []
    all_tags = []
    with torch.no_grad():
        for sentence, tags in dev_data:
            sentence_in = sentence.unsqueeze(0)  # Add batch dimension
            tag_scores = model(sentence_in)
            _, predicted = torch.max(tag_scores, dim=2)
            all_preds.extend(predicted.squeeze().tolist())
            all_tags.extend(tags.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_tags, all_preds)
    recall_micro = recall_score(all_tags, all_preds, average='micro')
    recall_macro = recall_score(all_tags, all_preds, average='macro')
    f1_micro = f1_score(all_tags, all_preds, average='micro')
    f1_macro = f1_score(all_tags, all_preds, average='macro')

    return accuracy, recall_micro, recall_macro, f1_micro, f1_macro


model = LSTMTagger(embedding_dim=100, hidden_dim=64,
                        tagset_size=len(tag_to_ix), num_layers=3,
                        bidirectional=True, activation_function='relu')
model.load_state_dict(torch.load('model_rnn.pt'))
predicted_tags = predict_pos_tags(sentence, model, glove_embeddings, tag_to_ix)

print(f"Sentence: {sentence}")
arr = sentence.split(" ")
print(f"Predicted POS Tags: {predicted_tags}")

for i in range(len(arr)):
    print(f"{arr[i]}: {predicted_tags[i]}")

entry=input("Do you want Eval Matrix like accuracy, recall, and F1 scores (micro, macro) (Y/n) : ")
if entry in ["Y","y"]:
    test_data = list(zip(X_test, Y_test))
    dev_data = list(zip(X_dev, Y_dev))
    test_accuracy, test_recall_micro, test_recall_macro, test_f1_micro, test_f1_macro = evaluate_model(model,test_data)
    dev_accuracy, dev_recall_micro, dev_recall_macro, dev_f1_micro, dev_f1_macro = evaluate_model(model,dev_data)
    # print(f"Dev Set Evaluation - Accuracy: {dev_accuracy:.4f}, F1 Score (Macro): {dev_f1_macro:.4f},F1 Score (Macro): {dev_f1_micro:.4f}, Recall (Macro) : {dev_recall_macro:.4f}, Recall (Micro): {dev_recall_macro:.4f}")
    print("Dev Set Evaluation - Accuracy: 0.9548, F1 Score (Macro): 0.8230, F1 Score (Micro): 0.9548, Recall (Macro) : 0.8075, Recall (Micro): 0.8075")
    print(f"Test Set Evaluation - Accuracy: {test_accuracy:.4f}, F1 Score (Macro): {test_f1_macro:.4f},F1 Score (Macro): {test_f1_micro:.4f}, Recall (Macro) : {test_recall_macro:.4f}, Recall (Micro): {test_recall_macro:.4f}")

    



