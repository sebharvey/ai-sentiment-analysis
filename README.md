# Sentiment Analysis Neural Network in C#

## Overview

This is a complete implementation of a feedforward neural network in C# that performs sentiment analysis on text. The network classifies sentences as **Positive**, **Negative**, or **Neutral** using a simple but effective architecture trained via backpropagation.

## Architecture

The neural network consists of three layers:

- **Input Layer**: 10 neurons (representing extracted features from text)
- **Hidden Layer**: 8 neurons with sigmoid activation
- **Output Layer**: 3 neurons (representing Positive, Negative, and Neutral classifications)
```
[10 Input Features] → [8 Hidden Neurons] → [3 Output Classes]
```

## How It Works

### 1. Feature Extraction

Before feeding text into the neural network, we need to convert it into numerical features. The system extracts 10 different features from each sentence:

| Feature | Description |
|---------|-------------|
| **Feature 1** | Average sentiment score from word dictionary |
| **Feature 2** | Maximum positive sentiment found in sentence |
| **Feature 3** | Minimum sentiment (most negative) in sentence |
| **Feature 4** | Count of positive words (normalized) |
| **Feature 5** | Count of negative words (normalized) |
| **Feature 6** | Normalized sentence length |
| **Feature 7** | Presence of exclamation marks (1.0 or 0.0) |
| **Feature 8** | Presence of question marks (1.0 or 0.0) |
| **Feature 9** | Ratio of sentiment words to total words |
| **Feature 10** | Variance in sentiment scores across the sentence |

#### Sentiment Dictionary

The system uses a pre-built dictionary that maps words to sentiment scores:

- **Positive words** (e.g., "love", "amazing", "brilliant") → scores between 0.6 and 1.0
- **Negative words** (e.g., "hate", "terrible", "worst") → scores between -1.0 and -0.6
- **Neutral words** (e.g., "okay", "average") → scores near 0.0

### 2. Neural Network Structure

#### Forward Propagation

The forward pass moves data through the network:

1. **Input → Hidden Layer**:
   - Each hidden neuron calculates: `sum = bias + Σ(input[i] × weight[i,j])`
   - Apply sigmoid activation: `hidden[j] = sigmoid(sum)`

2. **Hidden → Output Layer**:
   - Each output neuron calculates: `sum = bias + Σ(hidden[j] × weight[j,k])`
   - Apply sigmoid activation: `output[k] = sigmoid(sum)`

The **sigmoid function** is defined as:
```
sigmoid(x) = 1 / (1 + e^(-x))
```

This maps any value to a range between 0 and 1, which is perfect for probability-like outputs.

#### Backpropagation

The backward pass updates the network's weights based on errors:

1. **Calculate Output Error**:
   - For each output neuron: `error = expected - actual`
   - Calculate delta: `delta = error × sigmoid_derivative(output)`

2. **Calculate Hidden Layer Error**:
   - Propagate error backwards through weights
   - Calculate delta: `delta = error × sigmoid_derivative(hidden)`

3. **Update Weights**:
   - `new_weight = old_weight + learning_rate × delta × activation`
   - This is gradient descent in action

The **sigmoid derivative** is:
```
sigmoid_derivative(x) = x × (1 - x)
```

This tells us how much to adjust weights based on the error.

### 3. Training Process

Training happens in epochs (complete passes through the training data):

1. **Initialize** weights randomly between -0.5 and 0.5
2. For each epoch:
   - For each training example:
     - Extract features from the sentence
     - Forward pass: get network's prediction
     - Calculate error between prediction and expected output
     - Backward pass: update weights to reduce error
3. Repeat for 1000 epochs with a learning rate of 0.1

#### Training Data Format

Each training example consists of:
- A sentence (string)
- Expected output (array of 3 numbers)
  - `[1, 0, 0]` = Positive
  - `[0, 1, 0]` = Negative
  - `[0, 0, 1]` = Neutral

Example:
```csharp
("I love this it's amazing", new double[] { 1, 0, 0 })  // Positive
("I hate this terrible product", new double[] { 0, 1, 0 })  // Negative
("It's okay nothing special", new double[] { 0, 0, 1 })  // Neutral
```

### 4. Making Predictions

To analyse sentiment:

1. Extract the 10 features from the input sentence
2. Run a forward pass through the trained network
3. Get the output values from the 3 output neurons
4. Choose the neuron with the highest activation
5. Map to sentiment:
   - Neuron 0 highest → "Positive"
   - Neuron 1 highest → "Negative"
   - Neuron 2 highest → "Neutral"

## Code Structure

### Classes

#### `Program`
The entry point that demonstrates the network in action. Creates a `SentimentAnalyser`, trains it, and tests it on sample sentences.

#### `SentimentAnalyser`
The high-level interface for sentiment analysis. Handles:
- Initializing the neural network
- Building the sentiment word dictionary
- Training the network with sample data
- Extracting features from sentences
- Providing the simple `AnalyseSentiment(string)` method

#### `NeuralNetwork`
The core neural network implementation. Handles:
- Weight and bias initialization
- Forward propagation through layers
- Backpropagation and weight updates
- Sigmoid activation function

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Input Size** | 10 | Number of features extracted from text |
| **Hidden Size** | 8 | Number of neurons in hidden layer |
| **Output Size** | 3 | Number of sentiment classes |
| **Learning Rate** | 0.1 | How much to adjust weights each step |
| **Epochs** | 1000 | Number of complete training passes |
| **Random Seed** | 42 | For reproducible results |

## Usage Example
```csharp
// Create and train the analyser
var analyser = new SentimentAnalyser();
analyser.Train();

// Analyse a sentence
string result = analyser.AnalyseSentiment("I love this product!");
// Returns: "Positive"

result = analyser.AnalyseSentiment("This is terrible");
// Returns: "Negative"

result = analyser.AnalyseSentiment("It's okay");
// Returns: "Neutral"
```

## Limitations and Considerations

### Current Limitations

1. **Small Vocabulary**: The sentiment dictionary only contains about 25 words. Real-world applications would need thousands of words.

2. **Limited Training Data**: Only 14 training examples. Production systems use thousands or millions of examples.

3. **Simple Features**: More sophisticated features (n-grams, word embeddings, context) would improve accuracy.

4. **No Context Understanding**: The network doesn't understand word order, sarcasm, or complex sentence structures.

### Improvements for Production

To make this production-ready, consider:

1. **Larger Dataset**: Train on datasets like IMDB reviews, Twitter sentiment, or Amazon reviews
2. **Better Features**: Use word embeddings (Word2Vec, GloVe) or character-level features
3. **Deeper Network**: Add more hidden layers for better representation learning
4. **Advanced Architectures**: Consider RNNs, LSTMs, or Transformers for sequence understanding
5. **Regularization**: Add dropout or L2 regularization to prevent overfitting
6. **Validation Set**: Split data into training/validation/test sets
7. **Cross-Validation**: Test performance more rigorously

## Mathematical Foundation

### Sigmoid Function

The sigmoid function squashes values to the range (0, 1):
```
σ(x) = 1 / (1 + e^(-x))
```

Properties:
- σ(0) = 0.5
- σ(+∞) → 1
- σ(-∞) → 0
- Derivative: σ'(x) = σ(x) × (1 - σ(x))

### Gradient Descent

Weight updates follow the rule:
```
w_new = w_old + α × δ × a
```

Where:
- α = learning rate
- δ = error signal (delta)
- a = activation from previous layer

### Error Calculation

Mean Squared Error (MSE) is used:
```
Error = Σ(expected - actual)²
```

## Performance

With 1000 training epochs on 14 examples:
- Training typically converges to low error (< 0.1)
- Simple positive/negative sentences are classified correctly
- Neutral sentences are more challenging due to less distinct features
 
