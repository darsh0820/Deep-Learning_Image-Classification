# Neural Network with Batch Normalization

This project demonstrates a simple feedforward neural network with batch normalization implemented using TensorFlow and Keras. The network is designed for a classification task with 784 input features and 10 output classes.

## Model Architecture

The model consists of the following layers:

1. **Input Layer**:
   - Dense layer with 64 neurons and ReLU activation.
   - Batch normalization to stabilize and speed up training.

2. **Hidden Layer**:
   - Dense layer with 64 neurons and ReLU activation.
   - Batch normalization to further stabilize and speed up training.

3. **Output Layer**:
   - Dense layer with 10 neurons and softmax activation for multi-class classification.

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

## Acknowledgments
This project is built using TensorFlow and Keras, inspired by standard neural network practices including batch normalization for improving model performance.
