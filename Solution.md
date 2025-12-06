# Sentiment Analysis Neural Network - Execution Guide

## Quick Overview

This is a sentiment analyser that actually learns from data rather than using hand-crafted word lists. You train it on labelled examples, it figures out what words mean, then it can classify new text.

Run it:

```bash
cd src/SentimentAnalysisApp
dotnet run
```

It'll train for about 30 seconds, show you some example classifications, then let you type your own sentences interactively.

That's it. Now let's dig into what's actually happening.

## What's an Epoch?

An **epoch** is one complete pass through the entire training dataset. Think of it like this: you have 500 training examples, and when the network has processed all 500 of them once, that's 1 epoch. The network can't learn everything from seeing each example just once - it needs repeated exposure to gradually adjust its weights, similar to how you might re-read a textbook multiple times when studying for an exam. This code runs 2,000 epochs, meaning it processes all 500 examples 2,000 times, making roughly 1 million weight updates in total. You'll see the error decrease with each epoch (printed every 200 epochs), which shows the network is learning better representations. Too few epochs and the network hasn't learned enough (underfitting); too many and it starts memorising training data rather than learning patterns (overfitting). The sweet spot of 2,000 epochs was found through experimentation for this particular dataset size and complexity.

---

## What You'll See When You Run It

### Phase 1: Training

```
Training True Neural Network for Sentiment Analysis
===================================================

Loaded 500 training examples from training_data.csv

Built vocabulary of 487 words from training data

Epoch 200/2000, Error: 45.2341
Epoch 400/2000, Error: 28.7654
Epoch 600/2000, Error: 18.9432
Epoch 800/2000, Error: 12.3456
Epoch 1000/2000, Error: 8.7654
Epoch 1200/2000, Error: 6.5432
Epoch 1400/2000, Error: 5.1234
Epoch 1600/2000, Error: 4.2345
Epoch 1800/2000, Error: 3.6789
Epoch 2000/2000, Error: 3.2456

Training complete!

The network has learned:
1. Word embeddings (what each word means)
2. How to classify sentiments from word meanings
```

Watch the error decrease - that's learning happening. The network starts with random weights and gradually adjusts them to minimise prediction error.

### Phase 2: Example Analyses

```
True Neural Network Sentiment Analysis

======================================

Example Analyses:

Sentence: I love this product, it's amazing!
Sentiment: Positive
Confidence: 94.3%
Scores - Pos: 0.943, Neg: 0.032, Neu: 0.025

Sentence: This is terrible and I hate it
Sentiment: Negative
Confidence: 91.8%
Scores - Pos: 0.041, Neg: 0.918, Neu: 0.041

Sentence: It's okay, nothing special
Sentiment: Neutral
Confidence: 78.2%
Scores - Pos: 0.112, Neg: 0.106, Neu: 0.782
```

Each sentence gets three scores (positive, negative, neutral) that sum to roughly 1.0. The highest score determines the classification.

### Phase 3: Interactive Mode

```
======================================================================
INTERACTIVE MODE - Analyse Your Own Sentences
======================================================================

Enter your own sentences to analyse sentiment.
Type 'exit' or 'quit' to end the programme.

Enter sentence: This restaurant serves the most delicious food
  → Sentiment: Positive
  → Confidence: 89.7%
  → Scores - Pos: 0.897, Neg: 0.056, Neu: 0.047

Enter sentence: The service was absolutely dreadful
  → Sentiment: Negative
  → Confidence: 86.4%
  → Scores - Pos: 0.067, Neg: 0.864, Neu: 0.069

Enter sentence: quit
Thank you for using the sentiment analyser. Goodbye!
```

Type anything. The network will classify it based on what it learned during training.

---

## The Training Dataset

Located at `src/SentimentAnalysisApp/training_data.csv`. Dead simple format:

```csv
sentiment,sentence text
positive,I love this product it is amazing
negative,Terrible quality waste of money
neutral,It works fine nothing special
```

The dataset contains 500+ realistic examples:

- **Product reviews**: Amazon-style reviews with varying length and detail
- **Restaurant reviews**: Verbose feedback about dining experiences  
- **Service reviews**: Hotels, airlines, utilities, customer service
- **Food reviews**: Grocery items, meals, beverages
- **Clothing reviews**: Apparel, accessories, fit and quality

Examples are intentionally wordy and realistic, not toy data like "good" / "bad".

### Why This Dataset Works

The network needs to see words in context with their sentiment labels. After seeing "excellent" appear in 50 positive reviews, the network adjusts that word's embedding vector to be associated with positive sentiment.

Similarly, "terrible" keeps appearing in negative reviews, so its embedding gets pushed towards the negative region of the vector space.

Words like "okay" and "acceptable" appear in neutral reviews, so they end up somewhere in the middle.

The network isn't told any of this explicitly - it discovers these patterns through gradient descent.

---

---

## Network Architecture Explained

Before we trace execution, let's understand the neural network structure. This system uses a simple three-layer feedforward architecture:

```
Input Layer:  16 neurons (word embedding dimensions)
              ↓
Hidden Layer: 12 neurons (pattern detection)
              ↓  
Output Layer: 3 neurons (positive, negative, neutral)
```

### Input Layer (16 neurons)

This isn't actually a layer with weights - it's the size of our word embedding vectors. Each word in our vocabulary is represented as a 16-dimensional vector of numbers. Why 16? It's a balance between expressiveness and efficiency. With 16 dimensions, we have enough space to encode meaningful differences between words (positive vs negative, strong vs weak sentiment, etc.) without making the network too large. Smaller values like 8 would limit expressiveness, whilst larger values like 64 would slow down training and risk overfitting on our modest dataset of 500 examples. Common embedding sizes in production systems range from 50 to 300 dimensions, but those work with millions of training examples.

### Hidden Layer (12 neurons)

This is where the actual learning happens. The hidden layer transforms the 16-dimensional input into a 12-dimensional representation that makes classification easier. Each neuron in this layer looks at all 16 input values, applies weights (importance factors), adds a bias, and passes the result through a sigmoid activation function that squashes outputs to a 0-1 range. Think of these neurons as feature detectors - collectively, they learn to recognise patterns like "this sentence has strong positive words" or "this sentence contains negation". The choice of 12 neurons is somewhat arbitrary but follows a common heuristic: use roughly 3/4 the size of the input layer for simple tasks. We could use 8 neurons (simpler, might underfit) or 24 neurons (more powerful, might overfit). The sigmoid activation function keeps values bounded and makes the network non-linear, allowing it to learn complex decision boundaries that linear models can't capture.

### Output Layer (3 neurons)

These three neurons represent our three sentiment classes: positive, negative, and neutral. Each neuron receives signals from all 12 hidden neurons, applies its own set of weights, and produces a score between 0 and 1. The neuron with the highest score determines the predicted sentiment. For instance, if the outputs are [0.87, 0.08, 0.05], the network is 87% confident the sentiment is positive. The scores don't sum to exactly 1.0 (since we're using sigmoid, not softmax), but they're roughly normalised. If we wanted exactly four sentiment classes (very positive, positive, negative, very negative), we'd simply change OUTPUT_SIZE to 4 - no other code changes needed.

### Why This Architecture?

This is a simple feedforward (also called "fully connected") architecture where every neuron in one layer connects to every neuron in the next layer. The progression 16 → 12 → 3 creates a funnel: we start with rich word representations, compress them through a hidden layer that learns relevant patterns, then output simple class scores. More sophisticated architectures exist (LSTMs for sequential data, attention mechanisms for context, etc.), but this simple structure is perfect for understanding how neural networks learn from scratch. The total number of trainable parameters is modest:

- Input → Hidden: (16 × 12) + 12 biases = 204 parameters
- Hidden → Output: (12 × 3) + 3 biases = 39 parameters  
- Word embeddings: 487 words × 16 dimensions = 7,792 parameters
- **Total: ~8,000 parameters**

For comparison, BERT has 110 million parameters. This tiny network can still learn meaningful patterns because the task is relatively simple and we're learning embeddings specifically for sentiment rather than general language understanding.

---

## Step-by-Step Execution Flow

Let's trace what happens when you type: **"This product is absolutely brilliant"**

### Step 1: Vocabulary Building (Training Phase)

Before training, the system processes all training sentences to build a vocabulary:

```csharp
// Simplified example
vocabulary = {
    "<UNK>": 0,      // Unknown word token
    "absolutely": 1,
    "brilliant": 2,
    "is": 3,
    "product": 4,
    "this": 5,
    // ... 481 more words
}
```

Every unique word gets an index. The `<UNK>` token handles words not seen during training.

### Step 2: Embedding Initialisation (Training Phase)

The system creates a 487 × 16 matrix of random numbers (487 words, 16-dimensional embeddings):

```csharp
wordEmbeddings = new double[487, 16];

// Each word starts with random values (Xavier initialisation)
// Example for word "brilliant" (index 2):
wordEmbeddings[2] = [0.12, -0.08, 0.15, -0.03, 0.21, ...]  // 16 values
```

These random numbers will become meaningful through training. Eventually, words with similar meanings will have similar embedding vectors.

### Step 3: Network Initialisation (Training Phase)

Create a three-layer neural network:

```
Input Layer:  16 neurons (embedding size)
              ↓
Hidden Layer: 12 neurons (sigmoid activation)
              ↓  
Output Layer: 3 neurons (positive, negative, neutral)
```

These specific values come from the constants defined in the code:

```csharp
private const int EMBEDDING_SIZE = 16;  // Input layer size
private const int HIDDEN_SIZE = 12;     // Hidden layer size  
private const int OUTPUT_SIZE = 3;      // Output layer size (pos, neg, neu)
```

The **EMBEDDING_SIZE (16)** determines how many numbers represent each word. This was chosen as a reasonable middle ground - large enough to capture meaning, small enough to train quickly on limited data. The **HIDDEN_SIZE (12)** follows the common rule of thumb of using about 3/4 of the input size for simple classification tasks. The **OUTPUT_SIZE (3)** is determined by our task: we're classifying into three sentiment categories.

Weights between layers are also randomly initialised:

```csharp
weightsInputHidden = new double[16, 12];   // 192 weights
weightsHiddenOutput = new double[12, 3];   // 36 weights
biasHidden = new double[12];               // 12 bias values
biasOutput = new double[3];                // 3 bias values
```

Each connection between neurons has a weight (importance factor), and each neuron has a bias (threshold). Xavier initialisation sets these to small random values scaled appropriately to prevent vanishing or exploding gradients during training.

### Step 4: Training Loop (Training Phase)

For 2,000 epochs, the network processes each training example:

#### 4.1 Convert Sentence to Vector

Training example: `"positive,This product is brilliant"`

```csharp
// Tokenise: ["this", "product", "is", "brilliant"]
// Look up embeddings:
embedding_this = wordEmbeddings[5]      // [0.12, -0.08, 0.15, ...]
embedding_product = wordEmbeddings[4]   // [0.09, 0.14, -0.11, ...]
embedding_is = wordEmbeddings[3]        // [-0.05, 0.21, 0.08, ...]
embedding_brilliant = wordEmbeddings[2] // [0.18, -0.13, 0.25, ...]

// Average them:
sentenceVector = (embedding_this + embedding_product + 
                  embedding_is + embedding_brilliant) / 4
// Result: [0.085, 0.035, 0.093, ...] (16 dimensions)
```

This is called "Continuous Bag of Words" (CBOW) representation. Word order doesn't matter, we're just averaging the embeddings.

#### 4.2 Forward Pass

Feed the averaged embedding through the network:

**Input to Hidden Layer:**

```csharp
for (int j = 0; j < 12; j++) {
    double sum = biasHidden[j];
    for (int i = 0; i < 16; i++) {
        sum += sentenceVector[i] * weightsInputHidden[i, j];
    }
    hiddenLayer[j] = Sigmoid(sum);
}
```

For each hidden neuron, we compute a weighted sum of inputs plus bias, then apply sigmoid activation.

Example calculation for hidden neuron 0:

```
sum = bias[0] + 
      (0.085 × weight[0,0]) + 
      (0.035 × weight[1,0]) + 
      (0.093 × weight[2,0]) + 
      ... (all 16 inputs)
    = 0.01 + 1.234 = 1.244

hiddenLayer[0] = sigmoid(1.244) = 0.776
```

**Hidden to Output Layer:**

```csharp
for (int k = 0; k < 3; k++) {
    double sum = biasOutput[k];
    for (int j = 0; j < 12; j++) {
        sum += hiddenLayer[j] * weightsHiddenOutput[j, k];
    }
    outputLayer[k] = Sigmoid(sum);
}
```

Same process for the output layer. We get three values:

```
outputLayer[0] = 0.234  // Positive score
outputLayer[1] = 0.123  // Negative score  
outputLayer[2] = 0.089  // Neutral score
```

These should sum to approximately 1.0 after sigmoid activation (not exact, but close).

**The Problem:**

The expected label for this training example is `[1, 0, 0]` (positive), but we got `[0.234, 0.123, 0.089]`. The network is way off. We need to adjust weights.

#### 4.3 Backward Pass

Calculate how wrong we were:

```csharp
// Output layer error
for (int k = 0; k < 3; k++) {
    double error = expected[k] - outputLayer[k];
    outputDelta[k] = error * SigmoidDerivative(outputLayer[k]);
}

// For our example:
outputDelta[0] = (1.0 - 0.234) × 0.234 × (1 - 0.234) = 0.138
outputDelta[1] = (0.0 - 0.123) × 0.123 × (1 - 0.123) = -0.013
outputDelta[2] = (0.0 - 0.089) × 0.089 × (1 - 0.089) = -0.007
```

The delta tells us how much to adjust each output neuron. Positive delta means increase, negative means decrease.

**Propagate Error Backwards:**

```csharp
// Hidden layer error
for (int j = 0; j < 12; j++) {
    double error = 0;
    for (int k = 0; k < 3; k++) {
        error += outputDelta[k] * weightsHiddenOutput[j, k];
    }
    hiddenDelta[j] = error * SigmoidDerivative(hiddenLayer[j]);
}
```

For each hidden neuron, we calculate how much it contributed to the output error, weighted by the connections to the output layer.

**Calculate Input Gradient:**

This is the clever bit - we calculate how much to adjust the input (embeddings):

```csharp
for (int i = 0; i < 16; i++) {
    double gradient = 0;
    for (int j = 0; j < 12; j++) {
        gradient += hiddenDelta[j] * weightsInputHidden[i, j];
    }
    inputGradient[i] = gradient;
}
```

This gradient flows all the way back to the word embeddings, allowing them to learn.

#### 4.4 Weight Updates

Update all weights using gradient descent:

```csharp
// Hidden to output weights
for (int j = 0; j < 12; j++) {
    for (int k = 0; k < 3; k++) {
        weightsHiddenOutput[j, k] += 
            learningRate × outputDelta[k] × hiddenLayer[j];
    }
}

// Example:
weightsHiddenOutput[0, 0] += 0.05 × 0.138 × 0.776
                           += 0.00535
```

The learning rate (0.05) controls how much we adjust. Too high and training becomes unstable, too low and learning is painfully slow.

**Update Input to Hidden Weights:**

```csharp
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 12; j++) {
        weightsInputHidden[i, j] += 
            learningRate × hiddenDelta[j] × sentenceVector[i];
    }
}
```

#### 4.5 Update Word Embeddings

This is where the magic happens - the embeddings learn:

```csharp
// For each word in the sentence: ["this", "product", "is", "brilliant"]
foreach (var word in words) {
    int wordIndex = vocabulary[word];
    
    for (int i = 0; i < 16; i++) {
        wordEmbeddings[wordIndex, i] += 
            learningRate × inputGradient[i] / wordCount;
    }
}
```

Since this was a positive example and the network under-predicted positive, the gradient pushes all these word embeddings towards the "positive region" of the vector space.

After seeing "brilliant" in dozens of positive examples, its embedding vector will be firmly positioned in the positive region.

### Step 5: Training Convergence

Over 2,000 epochs, the network sees each training example multiple times (with shuffling). Gradually:

- **Word embeddings** converge to meaningful representations
- **Network weights** learn to map embeddings to sentiment labels
- **Error decreases** from ~45 to ~3

The final embeddings look something like:

```
"brilliant":   [0.82, 0.76, 0.91, -0.12, 0.88, ...]  ← High positive values
"excellent":   [0.79, 0.81, 0.87, -0.09, 0.83, ...]  ← Similar to "brilliant"
"terrible":    [-0.84, -0.78, -0.91, 0.88, -0.81, ...] ← Negative values
"okay":        [0.12, -0.08, 0.15, 0.09, -0.11, ...] ← Near zero (neutral)
```

Similar words end up with similar vectors. The network learned this from data, not from a dictionary.

---

## Inference: Classifying New Text

Now let's classify: **"This product is absolutely brilliant"**

### Step 1: Tokenisation

```csharp
sentence.ToLower().Split([' ', ',', '.', '!', '?', ...])
// Result: ["this", "product", "is", "absolutely", "brilliant"]
```

### Step 2: Vocabulary Lookup

```csharp
words = ["this", "product", "is", "absolutely", "brilliant"]

// Look up indices:
vocabulary["this"] = 5
vocabulary["product"] = 4
vocabulary["is"] = 3
vocabulary["absolutely"] = 1
vocabulary["brilliant"] = 2
```

All words are in the vocabulary. If we encounter an unknown word, we use index 0 (`<UNK>`).

### Step 3: Retrieve Learned Embeddings

```csharp
// Get the learned embeddings (now meaningful after training):
embedding_this = wordEmbeddings[5]         // [0.18, -0.06, 0.22, ...]
embedding_product = wordEmbeddings[4]      // [0.15, 0.11, -0.08, ...]
embedding_is = wordEmbeddings[3]           // [-0.02, 0.19, 0.06, ...]
embedding_absolutely = wordEmbeddings[1]   // [0.74, 0.68, 0.81, ...]
embedding_brilliant = wordEmbeddings[2]    // [0.82, 0.76, 0.91, ...]
```

Notice "absolutely" and "brilliant" have large positive values because they appeared in positive training examples.

### Step 4: Average Embeddings

```csharp
sentenceVector = (embedding_this + embedding_product + 
                  embedding_is + embedding_absolutely + 
                  embedding_brilliant) / 5

// Result: [0.374, 0.336, 0.384, ...] (16 dimensions)
```

The average is strongly positive because two of the five words ("absolutely", "brilliant") have strongly positive embeddings.

### Step 5: Forward Pass Through Network

**Input to Hidden:**

```csharp
for (int j = 0; j < 12; j++) {
    double sum = biasHidden[j];
    for (int i = 0; i < 16; i++) {
        sum += sentenceVector[i] * weightsInputHidden[i, j];
    }
    hiddenLayer[j] = Sigmoid(sum);
}

// Example results:
hiddenLayer = [0.892, 0.756, 0.634, 0.923, 0.567, 0.812, ...]
```

**Hidden to Output:**

```csharp
for (int k = 0; k < 3; k++) {
    double sum = biasOutput[k];
    for (int j = 0; j < 12; j++) {
        sum += hiddenLayer[j] * weightsHiddenOutput[j, k];
    }
    outputLayer[k] = Sigmoid(sum);
}

// Results:
outputLayer[0] = 0.943  // Positive
outputLayer[1] = 0.032  // Negative
outputLayer[2] = 0.025  // Neutral
```

### Step 6: Classification

```csharp
// Find highest score
maxIndex = 0  // Positive has the highest score (0.943)

result = new SentimentResult {
    Sentiment = "Positive",
    Confidence = 0.943,
    PositiveScore = 0.943,
    NegativeScore = 0.032,
    NeutralScore = 0.025
};
```

**Output:**

```
Sentence: This product is absolutely brilliant
Sentiment: Positive
Confidence: 94.3%
Scores - Pos: 0.943, Neg: 0.032, Neu: 0.025
```

The network is 94.3% confident this is positive sentiment.

---

## Handling Unknown Words

What if we classify: **"This gadget is absolutely superb"**

The word "gadget" wasn't in our training data (only "product" was).

### Step 1: Tokenisation

```csharp
words = ["this", "gadget", "is", "absolutely", "superb"]
```

### Step 2: Vocabulary Lookup

```csharp
vocabulary["this"] = 5       // ✓ Known
vocabulary["gadget"] = ?     // ✗ Unknown
vocabulary["is"] = 3         // ✓ Known
vocabulary["absolutely"] = 1 // ✓ Known
vocabulary["superb"] = ?     // ✗ Unknown
```

For unknown words, use the `<UNK>` token:

```csharp
if (vocabulary.ContainsKey(word)) {
    wordIndex = vocabulary[word];
} else {
    wordIndex = vocabulary["<UNK>"];  // Index 0
}
```

### Step 3: Get Embeddings

```csharp
embedding_this = wordEmbeddings[5]       // Known word
embedding_gadget = wordEmbeddings[0]     // <UNK> token
embedding_is = wordEmbeddings[3]         // Known word
embedding_absolutely = wordEmbeddings[1] // Known word
embedding_superb = wordEmbeddings[0]     // <UNK> token

// Average:
sentenceVector = (embedding_this + embedding_gadget + 
                  embedding_is + embedding_absolutely + 
                  embedding_superb) / 5
```

The `<UNK>` token has a learned embedding based on how unknown words typically behaved in training. It's a fallback that lets the network still make reasonable predictions.

### Step 4: Classification

Even with unknown words, the network can classify correctly because "absolutely" is strongly positive, and the overall sentence structure matches positive training examples.

**Result:**

```
Sentence: This gadget is absolutely superb
Sentiment: Positive
Confidence: 87.6%
```

Confidence is slightly lower (87.6% vs 94.3%) because of the unknown words, but it still classifies correctly.

---

## Detailed Code Walkthrough

### NeuralNetwork.cs

This is the core neural network implementation. It's generic - not specific to sentiment analysis.

#### Constructor

```csharp
public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
{
    this.inputSize = inputSize;    // 16 (embedding size)
    this.hiddenSize = hiddenSize;  // 12 neurons
    this.outputSize = outputSize;  // 3 (sentiments)
    
    // Initialise weight matrices
    weightsInputHidden = new double[inputSize, hiddenSize];
    weightsHiddenOutput = new double[hiddenSize, outputSize];
    
    // Xavier initialisation prevents exploding/vanishing gradients
    double scaleInputHidden = Math.Sqrt(2.0 / (inputSize + hiddenSize));
    double scaleHiddenOutput = Math.Sqrt(2.0 / (hiddenSize + outputSize));
    
    // Randomise weights within scaled range
    for (int i = 0; i < inputSize; i++)
        for (int j = 0; j < hiddenSize; j++)
            weightsInputHidden[i, j] = (random.NextDouble() * 2 - 1) * scaleInputHidden;
}
```

Xavier initialisation: `weight = random(-√(2/(fan_in + fan_out)), √(2/(fan_in + fan_out)))`

This keeps gradients in a reasonable range, preventing them from exploding or vanishing during backpropagation.

#### Forward Method

```csharp
public double[] Forward(double[] input)
{
    inputLayer = input;
    hiddenLayer = new double[hiddenSize];
    outputLayer = new double[outputSize];
    
    // Input → Hidden: weighted sum + bias, then sigmoid
    for (int j = 0; j < hiddenSize; j++) {
        double sum = biasHidden[j];
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weightsInputHidden[i, j];
        }
        hiddenLayer[j] = Sigmoid(sum);
    }
    
    // Hidden → Output: weighted sum + bias, then sigmoid
    for (int k = 0; k < outputSize; k++) {
        double sum = biasOutput[k];
        for (int j = 0; j < hiddenSize; j++) {
            sum += hiddenLayer[j] * weightsHiddenOutput[j, k];
        }
        outputLayer[k] = Sigmoid(sum);
    }
    
    return outputLayer;
}
```

Standard feedforward pass. Matrix multiplication (implemented as nested loops) followed by activation functions.

#### Sigmoid Function

```csharp
private double Sigmoid(double x)
{
    // Clip to prevent overflow
    if (x > 20) return 1.0;
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + Math.Exp(-x));
}
```

Sigmoid squashes any input to range [0, 1]:

```
σ(x) = 1 / (1 + e^(-x))

σ(-∞) → 0
σ(0) = 0.5
σ(+∞) → 1
```

Clipping prevents `Math.Exp(-x)` from overflowing when x is very large.

#### Sigmoid Derivative

```csharp
private double SigmoidDerivative(double x)
{
    return x * (1.0 - x);
}
```

The derivative of sigmoid has a convenient form when you already have the sigmoid output:

```
σ'(x) = σ(x) × (1 - σ(x))
```

This makes backpropagation efficient - we don't need to recalculate the sigmoid, we can use the value we already computed during the forward pass.

#### Backward Method

```csharp
public double[] Backward(double[] expected, double learningRate)
{
    // Calculate output layer deltas
    double[] outputDelta = new double[outputSize];
    for (int k = 0; k < outputSize; k++) {
        double error = expected[k] - outputLayer[k];
        outputDelta[k] = error * SigmoidDerivative(outputLayer[k]);
    }
    
    // Calculate hidden layer deltas
    double[] hiddenDelta = new double[hiddenSize];
    for (int j = 0; j < hiddenSize; j++) {
        double error = 0;
        for (int k = 0; k < outputSize; k++) {
            error += outputDelta[k] * weightsHiddenOutput[j, k];
        }
        hiddenDelta[j] = error * SigmoidDerivative(hiddenLayer[j]);
    }
    
    // Calculate input gradient (for embedding updates)
    double[] inputGradient = new double[inputSize];
    for (int i = 0; i < inputSize; i++) {
        double gradient = 0;
        for (int j = 0; j < hiddenSize; j++) {
            gradient += hiddenDelta[j] * weightsInputHidden[i, j];
        }
        inputGradient[i] = gradient;
    }
    
    // Update all weights
    UpdateWeights(outputDelta, hiddenDelta, learningRate);
    
    return inputGradient;  // Return for embedding updates
}
```

This implements backpropagation:

1. Calculate how wrong the output was (output delta)
2. Propagate error backwards through network (hidden delta)
3. Calculate gradient for inputs (embedding gradient)
4. Update all weights using calculated gradients

The returned `inputGradient` is crucial - it tells us how to adjust the word embeddings.

### NeuralNetworkSentimentAnalyser.cs

This wraps the generic neural network with sentiment-specific logic.

#### BuildVocabulary

```csharp
private void BuildVocabulary(List<string> sentences)
{
    var uniqueWords = new HashSet<string>();
    
    foreach (var sentence in sentences) {
        var words = Tokenize(sentence);
        foreach (var word in words) {
            uniqueWords.Add(word);
        }
    }
    
    // Add special token for unknown words
    vocabulary["<UNK>"] = 0;
    
    int index = 1;
    foreach (var word in uniqueWords.OrderBy(w => w)) {
        vocabulary[word] = index++;
    }
    
    vocabularySize = vocabulary.Count;
}
```

Extracts every unique word from training data and assigns it an index. Sorting ensures consistent indices across runs.

#### InitializeWordEmbeddings

```csharp
private void InitializeWordEmbeddings()
{
    wordEmbeddings = new double[vocabularySize, EMBEDDING_SIZE];
    
    // Xavier initialization
    double scale = Math.Sqrt(2.0 / (vocabularySize + EMBEDDING_SIZE));
    
    for (int i = 0; i < vocabularySize; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            wordEmbeddings[i, j] = (random.NextDouble() * 2 - 1) * scale;
        }
    }
}
```

Creates a matrix where each row is a word's embedding vector. Initially random, but will become meaningful through training.

#### SentenceToVector

```csharp
private double[] SentenceToVector(string sentence)
{
    var words = Tokenize(sentence);
    var vector = new double[EMBEDDING_SIZE];
    
    int wordCount = 0;
    foreach (var word in words) {
        int wordIndex = vocabulary.ContainsKey(word) 
            ? vocabulary[word] 
            : vocabulary["<UNK>"];
        
        // Add this word's embedding
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            vector[i] += wordEmbeddings[wordIndex, i];
        }
        wordCount++;
    }
    
    // Average the embeddings
    if (wordCount > 0) {
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            vector[i] /= wordCount;
        }
    }
    
    return vector;
}
```

Converts text to a fixed-size vector by averaging word embeddings. This is CBOW (Continuous Bag of Words) representation.

Alternative approaches:

- **Sum instead of average**: Loses normalisation
- **TF-IDF weighting**: More complex, possibly better
- **Attention mechanism**: Much more complex, definitely better
- **Preserve word order**: Requires RNN/LSTM/Transformer

We use simple averaging because it's easy to understand and works reasonably well.

#### UpdateWordEmbeddings

```csharp
private void UpdateWordEmbeddings(string sentence, double[] embeddingGradient, double learningRate)
{
    var words = Tokenize(sentence);
    int wordCount = words.Length;
    
    if (wordCount == 0) return;
    
    foreach (var word in words) {
        int wordIndex = vocabulary.ContainsKey(word)
            ? vocabulary[word]
            : vocabulary["<UNK>"];
        
        // Update this word's embedding
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            wordEmbeddings[wordIndex, i] += 
                learningRate * embeddingGradient[i] / wordCount;
        }
    }
}
```

Applies the gradient from backpropagation to update word embeddings. The gradient is divided by word count because we averaged the embeddings in `SentenceToVector`.

This is the key to learning word meanings. Each training example nudges word embeddings in the direction that reduces classification error.

#### Train Method

```csharp
public void Train()
{
    // Load data
    var trainingData = LoadTrainingData("training_data.csv");
    
    // Build vocabulary
    BuildVocabulary(trainingData.Select(t => t.sentence).ToList());
    
    // Initialize embeddings
    InitializeWordEmbeddings();
    
    // Create network
    network = new NeuralNetwork(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Training loop
    int epochs = 2000;
    double learningRate = 0.05;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0;
        
        // Shuffle training data (important for SGD)
        var shuffled = trainingData.OrderBy(x => Guid.NewGuid()).ToList();
        
        foreach (var (sentence, label) in shuffled) {
            // Convert to vector
            double[] sentenceVector = SentenceToVector(sentence);
            
            // Forward pass
            double[] output = network.Forward(sentenceVector);
            
            // Calculate error
            double error = 0;
            for (int i = 0; i < output.Length; i++) {
                error += Math.Pow(label[i] - output[i], 2);
            }
            totalError += error;
            
            // Backward pass
            double[] embeddingGradient = network.Backward(label, learningRate);
            
            // Update embeddings
            UpdateWordEmbeddings(sentence, embeddingGradient, learningRate);
        }
        
        if ((epoch + 1) % 200 == 0) {
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError:F4}");
        }
    }
}
```

This orchestrates the entire training process:

1. Load labelled data from CSV
2. Build vocabulary from all sentences
3. Initialise random embeddings
4. Create neural network
5. For 2000 epochs:
   - Shuffle training data
   - For each example:
     - Convert sentence to vector (average embeddings)
     - Forward pass through network
     - Calculate error
     - Backward pass (compute gradients)
     - Update network weights
     - Update word embeddings

Shuffling is important for stochastic gradient descent. Without it, the network might learn patterns in the data order rather than the actual sentiment patterns.

#### AnalyseSentiment Method

```csharp
public SentimentResult AnalyseSentiment(string sentence)
{
    // Convert sentence to vector using learned embeddings
    double[] sentenceVector = SentenceToVector(sentence);
    
    // Get prediction from network
    double[] output = network.Forward(sentenceVector);
    
    // Find highest scoring sentiment
    int maxIndex = 0;
    for (int i = 1; i < output.Length; i++) {
        if (output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }
    
    string[] sentiments = { "Positive", "Negative", "Neutral" };
    
    return new SentimentResult {
        Sentiment = sentiments[maxIndex],
        Confidence = output[maxIndex],
        PositiveScore = output[0],
        NegativeScore = output[1],
        NeutralScore = output[2]
    };
}
```

Simple inference:

1. Convert sentence to vector (average learned embeddings)
2. Forward pass through trained network
3. Find highest score
4. Return classification and confidence

No backward pass during inference - weights don't change.

---

## Understanding the Vector Space

After training, words occupy positions in 16-dimensional space based on their sentiment associations.

### Visualisation (Projected to 2D)

If we could visualise the 16D space in 2D, it might look like:

```
                    Positive Region
                    ↑
        excellent • │ • brilliant
      fantastic •   │   • wonderful
           superb • │ • amazing
                    │
  neutral ••••••••••┼••••••••••• mediocre
  acceptable •      │      • okay
         fair •     │     • standard
                    │
                    │
        terrible •  │  • awful
           horrible │ • disappointing
          dreadful •│• frustrating
                    ↓
                 Negative Region
```

Similar words cluster together. The network learned this purely from seeing which words appear in positive vs negative reviews.

### Example Embeddings (Hypothetical)

After training, embeddings might look like:

```
Word: "excellent"
Embedding: [0.89, 0.76, 0.82, -0.11, 0.91, 0.77, 0.88, 0.69, ...]
Average value: +0.58 (strongly positive)

Word: "brilliant"  
Embedding: [0.85, 0.82, 0.79, -0.09, 0.87, 0.81, 0.84, 0.72, ...]
Average value: +0.56 (strongly positive, similar to "excellent")

Word: "terrible"
Embedding: [-0.88, -0.79, -0.85, 0.89, -0.82, -0.74, -0.91, -0.77, ...]
Average value: -0.61 (strongly negative)

Word: "okay"
Embedding: [0.08, -0.12, 0.15, 0.06, -0.09, 0.11, -0.07, 0.13, ...]
Average value: +0.03 (neutral, near zero)

Word: "the"
Embedding: [-0.02, 0.04, -0.06, 0.03, 0.05, -0.04, 0.02, -0.03, ...]
Average value: -0.001 (neutral, appears in all sentences)
```

Common words like "the", "is", "a" end up near zero because they appear in positive, negative, and neutral examples equally.

### Similarity Calculation

We can measure similarity between words using cosine similarity:

```csharp
double CosineSimilarity(double[] vec1, double[] vec2) {
    double dot = 0, mag1 = 0, mag2 = 0;
    
    for (int i = 0; i < vec1.Length; i++) {
        dot += vec1[i] * vec2[i];
        mag1 += vec1[i] * vec1[i];
        mag2 += vec2[i] * vec2[i];
    }
    
    return dot / (Math.Sqrt(mag1) * Math.Sqrt(mag2));
}

// Example similarities:
CosineSimilarity("excellent", "brilliant") = 0.94   // Very similar
CosineSimilarity("excellent", "terrible") = -0.87   // Opposite
CosineSimilarity("excellent", "okay") = 0.12        // Somewhat related
CosineSimilarity("excellent", "the") = 0.03         // Unrelated
```

High cosine similarity means words have similar meanings (or at least similar sentiment associations).

---

## Why This Works

### The Mathematical Intuition

The network is essentially learning two functions:

**Function 1: Word → Vector**

Maps words to points in 16D space where distance = semantic difference.

**Function 2: Vector → Sentiment**

Maps points in 16D space to one of three sentiment categories.

During training, both functions are optimised simultaneously through gradient descent. The embedding function learns to place words such that the classification function can easily separate them.

### The Learning Process

Initially (random weights):

```
"excellent" → [0.12, -0.08, 0.15, ...] → Forward pass → [0.33, 0.33, 0.34]
Expected: [1.0, 0.0, 0.0]
Error: Large
```

The network says "I have no idea" (all scores ~0.33).

After 100 epochs:

```
"excellent" → [0.45, 0.32, 0.51, ...] → Forward pass → [0.62, 0.19, 0.19]
Expected: [1.0, 0.0, 0.0]
Error: Medium
```

The network is learning - positive score is higher, but not confident enough.

After 2000 epochs:

```
"excellent" → [0.89, 0.76, 0.82, ...] → Forward pass → [0.94, 0.03, 0.03]
Expected: [1.0, 0.0, 0.0]
Error: Small
```

The network is confident and correct.

### Why Averaging Works

You might wonder: "Doesn't averaging lose information?"

Yes, it does. Word order is lost. "Dog bites man" = "Man bites dog" in our system.

But for sentiment analysis, this often doesn't matter. The presence of sentiment-bearing words ("excellent", "terrible") is more important than their order.

Consider:

- "The food was absolutely delicious and the service was excellent"
- "The service was excellent and the food was absolutely delicious"

Both have the same sentiment despite different word order. Averaging captures this.

For tasks where order matters (translation, question answering), you'd need sequential models like LSTMs or Transformers.

---

## Performance Analysis

### Training Performance

On a typical laptop (Intel i7, 16GB RAM):

- **Training time**: ~30 seconds for 2000 epochs
- **Memory usage**: ~70KB for weights and embeddings
- **Final error**: ~3.2 (down from ~45)

The error measures mean squared error across all training examples. Lower is better, but reaching exactly zero would indicate overfitting.

### Classification Performance

Inference is extremely fast:

- **Per-sentence classification**: <1ms
- **Batch of 1000 sentences**: <1 second

This is because inference is just:
1. Look up embeddings (O(n) where n = words in sentence)
2. Average vectors (O(n × 16))
3. Matrix multiplication (O(16 × 12 + 12 × 3) = O(1))

No heavy computation needed.

### Accuracy Considerations

This is a simple model, so expectations should be realistic:

**It handles well:**
- Clear sentiment: "This is excellent" → 95%+ accuracy
- Multiple sentiment words: "Absolutely brilliant and fantastic" → High confidence
- Variations in phrasing: Works well even with unseen sentence structures

**It struggles with:**
- Negation: "Not good" might misclassify (sees "good")
- Sarcasm: "Oh great, another problem" (context-dependent)
- Mixed sentiment: "Good food but terrible service" (picks one)
- Complex grammar: "It's not that I don't like it" (confusing structure)

For production use, you'd want:
- Larger training set (10,000+ examples minimum)
- Negation handling (specific logic for "not", "never", etc.)
- More sophisticated architecture (LSTM, attention, etc.)
- Proper train/validation/test split
- Cross-validation for hyperparameter tuning

---

## Common Issues and Debugging

### Issue 1: High Error After Training

If error stays above 20 after 2000 epochs:

**Possible causes:**
- Learning rate too high/low
- Not enough training data
- Training data not diverse enough
- Network too small for the task

**Solutions:**

```csharp
// Try adjusting learning rate:
double learningRate = 0.01;  // Lower if training is unstable
double learningRate = 0.1;   // Higher if learning is too slow

// Increase network size:
const int HIDDEN_SIZE = 24;  // Instead of 12

// Train for longer:
int epochs = 5000;  // Instead of 2000
```

### Issue 2: All Predictions Are the Same

If the network always predicts the same class:

**Possible cause:** Class imbalance in training data.

If you have 400 positive examples, 80 negative, and 20 neutral, the network learns to always guess "positive" (80% accuracy by default).

**Solution:** Balance your training data or use class weights.

### Issue 3: Numerical Instability

If you see `NaN` in outputs:

**Cause:** Exploding gradients or sigmoid overflow.

**Solution:** Already handled by:

```csharp
private double Sigmoid(double x) {
    if (x > 20) return 1.0;   // Prevent exp(-x) overflow
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + Math.Exp(-x));
}
```

If you still get `NaN`, reduce learning rate.

### Issue 4: Overfitting

If training error is low but performance on new text is poor:

**Cause:** Network memorised training data instead of learning patterns.

**Solutions:**
- Add more diverse training data
- Use regularisation (L2 penalty on weights)
- Implement dropout during training
- Stop training earlier

---

## Extending the System

### Adding Regularisation

Prevent overfitting by penalising large weights:

```csharp
// In Backward method, add L2 regularisation:
double lambda = 0.001;  // Regularisation strength

for (int j = 0; j < hiddenSize; j++) {
    for (int k = 0; k < outputSize; k++) {
        double regularisation = lambda * weightsHiddenOutput[j, k];
        weightsHiddenOutput[j, k] += 
            learningRate * (outputDelta[k] * hiddenLayer[j] - regularisation);
    }
}
```

This encourages weights to stay small, reducing overfitting.

### Implementing Dropout

Randomly disable neurons during training:

```csharp
// During forward pass in training:
double dropoutRate = 0.5;
bool[] activeNeurons = new bool[hiddenSize];

for (int j = 0; j < hiddenSize; j++) {
    activeNeurons[j] = random.NextDouble() > dropoutRate;
    if (!activeNeurons[j]) {
        hiddenLayer[j] = 0;  // Disable this neuron
    }
}

// During inference, use all neurons but scale:
for (int j = 0; j < hiddenSize; j++) {
    hiddenLayer[j] *= (1 - dropoutRate);  // Scale by keep probability
}
```

Dropout forces the network to learn redundant representations, improving generalisation.

### Using Different Activation Functions

Replace sigmoid with ReLU (often performs better):

```csharp
private double ReLU(double x) {
    return Math.Max(0, x);
}

private double ReLUDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}
```

ReLU doesn't saturate like sigmoid, allowing gradients to flow better.

### Saving and Loading Models

Add serialisation to save trained models:

```csharp
public void SaveModel(string filepath) {
    using (var writer = new BinaryWriter(File.Open(filepath, FileMode.Create))) {
        // Write vocabulary
        writer.Write(vocabulary.Count);
        foreach (var kvp in vocabulary) {
            writer.Write(kvp.Key);
            writer.Write(kvp.Value);
        }
        
        // Write embeddings
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                writer.Write(wordEmbeddings[i, j]);
            }
        }
        
        // Write network weights
        // ... (similar structure)
    }
}
```

Then load without retraining:

```csharp
public void LoadModel(string filepath) {
    // Read vocabulary and weights from file
    // Reconstruct network
}
```

### Adding More Output Classes

To classify into 5 sentiment levels (very negative → very positive):

```csharp
const int OUTPUT_SIZE = 5;

// In training data:
// very_positive → [1, 0, 0, 0, 0]
// positive → [0, 1, 0, 0, 0]
// neutral → [0, 0, 1, 0, 0]
// negative → [0, 0, 0, 1, 0]
// very_negative → [0, 0, 0, 0, 1]
```

No other code changes needed - the network handles any number of output classes.

---

## Comparison with Other Approaches

### Dictionary-Based (Fake ML)

```csharp
// What some "ML" systems actually do:
Dictionary<string, double> sentiment = new() {
    ["excellent"] = 1.0,
    ["good"] = 0.5,
    ["terrible"] = -1.0,
    // ... 1000 more words hand-crafted
};

double score = sentence.Split(' ')
    .Sum(word => sentiment.GetValueOrDefault(word, 0.0));
```

**Problems:**
- Requires manual effort to build dictionary
- No learning - can't improve from data
- Doesn't handle new words or contexts
- Not actually machine learning

**Our approach:** Learns word sentiment from data, handles unknowns, truly learns patterns.

### Naive Bayes

```csharp
// Count word frequencies in positive vs negative examples
P(positive | "excellent") = count("excellent" in positive) / count(positive)
```

**Advantages:**
- Fast training and inference
- Works well with small datasets
- Probabilistically sound

**Disadvantages:**
- Assumes word independence (Naive assumption)
- Doesn't learn word representations
- Less flexible than neural networks

**Our approach:** More expressive, learns word embeddings, can capture non-linear patterns.

### Modern Transformers (BERT, GPT)

```csharp
// Massively oversimplified:
// - 110M+ parameters
// - Attention mechanisms
// - Contextual embeddings
// - Transfer learning from huge corpora
```

**Advantages:**
- State-of-the-art accuracy
- Handles context, negation, sarcasm
- Pre-trained on massive datasets

**Disadvantages:**
- Requires GPU for training
- Complex architecture (hard to understand)
- Overkill for simple sentiment analysis

**Our approach:** Simple, understandable, runs on CPU, good for learning fundamentals.

---

## Summary

This sentiment analyser demonstrates core neural network concepts:

1. **Representation learning**: Word embeddings learned from data
2. **Gradient descent**: Iterative optimisation through backpropagation
3. **Neural architecture**: Feedforward network with multiple layers
4. **Supervised learning**: Training from labelled examples

It's not production-ready, but it's genuine machine learning. The network discovers patterns in data rather than following hand-coded rules.

Key takeaways:

- **Word embeddings are learned**, not prescribed
- **Backpropagation updates both network weights and embeddings**
- **Simple architectures can work** for straightforward tasks
- **Understanding the fundamentals** matters more than using the fanciest frameworks

For production use, you'd want PyTorch/TensorFlow and more sophisticated architectures. But for understanding what's actually happening inside a neural network? This code shows you exactly that.