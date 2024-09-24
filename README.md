# Deep Learning Algorithms (Image Classification)

This repository provides an overview of various deep learning algorithms for image classification, focusing on their structures, use cases, and implementation in Python using TensorFlow/Keras. The models discussed include:

1. **Multilayer Perceptron (MLP)**
2. **Convolutional Neural Networks (CNN)**
3. **Recurrent Neural Networks (RNN)**

## Batch Normalization
Batch normalization is used to normalize the outputs of the Dense layers. This helps in:
- Reducing internal covariate shift
- Speeding up the training process
- Reducing the sensitivity to initialization
- Improving generalization
  
## Requirements
1. Python 3
2. TensorFlow
3. Keras
4. NumPy

## 1. Multilayer Perceptron (MLP)
### Overview
The Multilayer Perceptron (MLP) is a fundamental class of feedforward artificial neural networks. An MLP consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

### Architecture
- **Input Layer**: Takes in input data (e.g., flattened image pixels).
- **Hidden Layers**: Dense layers with neurons that learn to recognize patterns from the input data.
- **Output Layer**: Typically uses a softmax function for classification tasks.

### Use Cases
- MLPs are used for tasks where the input data is not structured in a grid-like format, such as:
  - Classification of tabular data
  - Simple regression tasks
  - Handwritten digit recognition (with flattened input)

### Example
```python
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## 2. Convolutional Neural Networks (CNN)
### Overview
Convolutional Neural Networks (CNNs) are specialized neural networks for processing data that has a known grid-like topology, such as images. CNNs are particularly effective at recognizing patterns, such as edges, textures, and objects, in visual data.

### Architecture
- **Convolutional Layers (Conv2D)**: Apply filters to the input data to extract features.
- **Pooling Layers (MaxPooling2D)**: Reduce the dimensionality of feature maps, retaining essential information.
- **Fully Connected Layers (Dense)**: After convolutional layers, one or more dense layers are used for classification or regression.

### Use Cases
- CNNs are widely used for image-related tasks, such as:
   - Image classification (e.g., MNIST digit recognition)
   - Object detection
   - Image segmentation

### Example
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## 3. Recurrent Neural Networks (RNN)
### Overview
Recurrent Neural Networks (RNNs) are designed for sequential data where the order of the input data is important. RNNs maintain a memory of previous inputs, allowing them to capture temporal dependencies in sequences.

### Architecture
- **LSTM/GRU Layers**: Special types of RNN layers that are capable of learning long-term dependencies in sequence data.
- **Dense Layers**: After the sequence processing, one or more dense layers are used for final predictions.
### Use Cases
- RNNs are ideal for tasks involving sequences, such as:
   - Time series forecasting
   - Natural language processing (NLP)
   - Sequence-to-sequence models (e.g., translation, text generation)

### Example
```python
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(28, 28)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## 4. Long Short-Term Memory (LSTM)
### Overview
Long Short-Term Memory (LSTM) networks are a special kind of RNN capable of learning long-term dependencies in sequences. Traditional RNNs can suffer from problems like vanishing and exploding gradients, making it difficult for them to learn long-range dependencies. LSTMs address these issues by learning which data is important to retain and which can be discarded, effectively managing information over longer sequences.

### Architecture
- **LSTM/GRU Layers**: Special types of RNN layers that are capable of learning long-term dependencies in sequence data.
- **Dense Layers**: After the sequence processing, one or more dense layers are used for final predictions.

### Use Cases
- RNNs like LSTMs are useful for tasks involving sequences where the order of the input matters. Common use cases include:
  - Time series forecasting
  - Natural Language Processing (NLP)
  - Sequence-to-sequence models (e.g., translation, text generation)
 
### Example
```python
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(28, 28), activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## Acknowledgments
This project is built using TensorFlow and Keras, inspired by standard neural network practices including batch normalization for improving model performance.
