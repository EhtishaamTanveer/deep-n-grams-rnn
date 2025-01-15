# Deep-N grams - RNNs

This repository contains a deep learning-based model for text generation using Recurrent Neural Networks (RNNs), specifically Gated Recurrent Units (GRUs). The model generates text by learning from a given text dataset. The key aspects of the project include:
1. Preprocessing the data
2. Defining a GRU-based language model
3. Training the model
4. Evaluating it using perplexity.
5. Additionally, the model is capable of generating text after training.

## Introduction

This project explores the use of GRUs (Gated Recurrent Units) for text generation. The model works by processing a given corpus of text, learning patterns within it, and then using this knowledge to generate new, contextually relevant text based on the input it receives. Perplexity, a commonly used metric in natural language processing (NLP) tasks, is used to evaluate the model’s performance. The entire process from data preprocessing to text generation is covered in this project.


## Project Structure

deep_n_grams_rnn/

|

├── data/               

│   └── shakespeare_data.txt  

├── deep-n-grams-rnn.ipynb  

├── utils.py 

├── test_utils.py 

├── w1_unittest.py

├── README.md           

## Requirements

To run this project, get the libraries through the command below:

```
pip install os traceback numpy random tensorflow termcolor string re nltk 
```

## Installation
1. Clone this repository:
```
git clone https://github.com/your_username/deep_n_grams_rnn.git
```
2. Open the main jupyter notebook called in your local environment i.e. VSCODE:
```
deep-n-ngrams.ipynb
```
3. Run the notebook to execute the project code
4. If you want to place your own dataset, keep it inside the data/folder.

## Usage
### Data Preprocessing

- Before training the model, you need to preprocess the text data. The following steps will help you convert the raw text into a suitable format for training:

1. **Load the dataset:** Load your raw text dataset (e.g., .txt file containing sentences).
2. **Create a vocabulary:** Tokenize the dataset into words or characters and create a vocabulary from the unique tokens.
3. **Convert text to tensor:** Map each word or character to a corresponding integer (index) from the vocabulary. Then, convert this representation into tensors for training.

### Model Training

- The core part of the project is the GRU-based language model. You can train the model using TensorFlow by defining the model architecture and training it on the preprocessed data.

1. **Define the GRU model**
2. **Train the model** 

### Evaluation

- To evaluate the model's performance, we use perplexity as the evaluation metric, which measures how well the model predicts the next word in a sequence.
- Calculate perplexity: After training, calculate the perplexity to assess how well the model has learned the text patterns.

Why Perplexity?

- Lower perplexity indicates that the model is more confident and accurate in its predictions.
- A higher perplexity suggests that the model is less accurate and struggles with generating coherent sequences.

### Text Generation

- Once the model is trained, it can generate new text based on a given seed or starting phrase. You can feed the model a starting sequence and let it predict the next words iteratively.

## Model Details

The model architecture is based on a GRU (Gated Recurrent Unit) network, which is a variant of LSTMs (Long Short-Term Memory units). The GRU architecture helps mitigate the vanishing gradient problem, making it a suitable choice for training on sequential data like text.

- Model Layers:

1. Embedding Layer: Maps input tokens (words/characters) to dense vectors of fixed size.
2. GRU Layer: Processes sequences by capturing long-range dependencies in the text.
3. Dense Layer: Outputs a probability distribution for the next token.

- Hyperparameters:

1. embedding_dim: Dimensionality of the embedding space (e.g., 256).
2. hidden_units: Number of units in the GRU layer (e.g., 512).
3. vocab_size: Size of the vocabulary (number of unique tokens in the dataset).

## Acknowledgements

- Thanks to TensorFlow for providing the necessary tools and models to implement and train the GRU-based language model.
- Special thanks to Coursera, for providing the dataset.

