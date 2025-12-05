# Sentiment Analysis Neural Network - Architecture Diagram

## Complete System Flow

```mermaid
flowchart TB
    Start([Input: "I love this product!"]) --> Extract[Feature Extraction]
    
    Extract --> F1[Feature 1: Avg Sentiment<br/>Score: 0.8]
    Extract --> F2[Feature 2: Max Positive<br/>Score: 1.0]
    Extract --> F3[Feature 3: Min Sentiment<br/>Score: 0.0]
    Extract --> F4[Feature 4: Positive Words<br/>Count: 0.2]
    Extract --> F5[Feature 5: Negative Words<br/>Count: 0.0]
    Extract --> F6[Feature 6: Length<br/>Score: 0.2]
    Extract --> F7[Feature 7: Exclamation<br/>Score: 1.0]
    Extract --> F8[Feature 8: Question Mark<br/>Score: 0.0]
    Extract --> F9[Feature 9: Sentiment Ratio<br/>Score: 0.5]
    Extract --> F10[Feature 10: Variance<br/>Score: 0.0]
    
    F1 --> NN[Neural Network]
    F2 --> NN
    F3 --> NN
    F4 --> NN
    F5 --> NN
    F6 --> NN
    F7 --> NN
    F8 --> NN
    F9 --> NN
    F10 --> NN
    
    NN --> O1[Output 1: Positive<br/>Score: 0.92]
    NN --> O2[Output 2: Negative<br/>Score: 0.05]
    NN --> O3[Output 3: Neutral<br/>Score: 0.12]
    
    O1 --> Decision{Find Highest<br/>Output}
    O2 --> Decision
    O3 --> Decision
    
    Decision --> Result([Result: "Positive"])
    
    style Start fill:#e1f5ff
    style Result fill:#90EE90
    style NN fill:#FFE5B4
    style Extract fill:#FFD700
    style Decision fill:#DDA0DD
```

## Neural Network Internal Architecture

```mermaid
graph LR
    subgraph Input["Input Layer (10 neurons)"]
        I1((F1))
        I2((F2))
        I3((F3))
        I4((F4))
        I5((F5))
        I6((F6))
        I7((F7))
        I8((F8))
        I9((F9))
        I10((F10))
    end
    
    subgraph Hidden["Hidden Layer (8 neurons)"]
        H1((H1))
        H2((H2))
        H3((H3))
        H4((H4))
        H5((H5))
        H6((H6))
        H7((H7))
        H8((H8))
    end
    
    subgraph Output["Output Layer (3 neurons)"]
        O1((Positive))
        O2((Negative))
        O3((Neutral))
    end
    
    I1 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I2 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I3 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I4 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I5 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I6 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I7 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I8 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I9 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    I10 -.-> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    
    H1 ==> O1 & O2 & O3
    H2 ==> O1 & O2 & O3
    H3 ==> O1 & O2 & O3
    H4 ==> O1 & O2 & O3
    H5 ==> O1 & O2 & O3
    H6 ==> O1 & O2 & O3
    H7 ==> O1 & O2 & O3
    H8 ==> O1 & O2 & O3
    
    style Input fill:#e1f5ff
    style Hidden fill:#FFE5B4
    style Output fill:#90EE90
```

## Training Process Flow

```mermaid
flowchart TD
    Start([Start Training]) --> Init[Initialize Network<br/>Random weights & biases]
    Init --> Epoch{Epoch < 1000?}
    
    Epoch -->|Yes| Sample[Get Training Sample]
    Sample --> FwdPass[Forward Pass<br/>Calculate predictions]
    FwdPass --> CalcError[Calculate Error<br/>MSE between expected<br/>and actual output]
    CalcError --> BackProp[Backpropagation<br/>Calculate gradients]
    BackProp --> UpdateW[Update Weights<br/>w = w + α × δ × a]
    UpdateW --> MoreSamples{More samples?}
    
    MoreSamples -->|Yes| Sample
    MoreSamples -->|No| PrintError[Print Epoch Error]
    PrintError --> Epoch
    
    Epoch -->|No| Complete([Training Complete])
    
    style Start fill:#e1f5ff
    style Complete fill:#90EE90
    style BackProp fill:#FFB6C1
    style UpdateW fill:#DDA0DD
```

## Forward Propagation Detail

```mermaid
flowchart LR
    subgraph Input["Input (10 features)"]
        I[Input Vector<br/>x₁, x₂, ..., x₁₀]
    end
    
    subgraph Hidden["Hidden Layer Calculation"]
        W1[Weights Matrix<br/>W⁽¹⁾ [10×8]]
        B1[Bias Vector<br/>b⁽¹⁾ [8]]
        Sum1[Σ = W⁽¹⁾·x + b⁽¹⁾]
        Act1[Sigmoid<br/>h = σ(Σ)]
    end
    
    subgraph Output["Output Layer Calculation"]
        W2[Weights Matrix<br/>W⁽²⁾ [8×3]]
        B2[Bias Vector<br/>b⁽²⁾ [3]]
        Sum2[Σ = W⁽²⁾·h + b⁽²⁾]
        Act2[Sigmoid<br/>y = σ(Σ)]
    end
    
    subgraph Result["Result"]
        Y[Output Vector<br/>y₁, y₂, y₃]
    end
    
    I --> W1
    W1 --> Sum1
    B1 --> Sum1
    Sum1 --> Act1
    Act1 --> W2
    W2 --> Sum2
    B2 --> Sum2
    Sum2 --> Act2
    Act2 --> Y
    
    style Input fill:#e1f5ff
    style Hidden fill:#FFE5B4
    style Output fill:#FFD700
    style Result fill:#90EE90
```

## Backpropagation Detail

```mermaid
flowchart RL
    subgraph Expected["Expected Output"]
        E[Target Vector<br/>t₁, t₂, t₃]
    end
    
    subgraph OutputError["Output Layer Error"]
        Err[Error = t - y]
        Delta2[δ⁽²⁾ = Error × σ'(y)]
    end
    
    subgraph UpdateOutput["Update Output Weights"]
        UpdateW2[W⁽²⁾ += α × δ⁽²⁾ × h]
        UpdateB2[b⁽²⁾ += α × δ⁽²⁾]
    end
    
    subgraph HiddenError["Hidden Layer Error"]
        BackErr[Error = δ⁽²⁾ · W⁽²⁾ᵀ]
        Delta1[δ⁽¹⁾ = Error × σ'(h)]
    end
    
    subgraph UpdateHidden["Update Hidden Weights"]
        UpdateW1[W⁽¹⁾ += α × δ⁽¹⁾ × x]
        UpdateB1[b⁽¹⁾ += α × δ⁽¹⁾]
    end
    
    E --> Err
    Err --> Delta2
    Delta2 --> UpdateW2
    Delta2 --> UpdateB2
    Delta2 --> BackErr
    BackErr --> Delta1
    Delta1 --> UpdateW1
    Delta1 --> UpdateB1
    
    style Expected fill:#FFB6C1
    style OutputError fill:#DDA0DD
    style UpdateOutput fill:#FFD700
    style HiddenError fill:#DDA0DD
    style UpdateHidden fill:#FFD700
```

## Feature Extraction Detail

```mermaid
flowchart TD
    Input["Input Sentence:<br/>'I love this amazing product!'"] --> Parse[Parse & Tokenize]
    
    Parse --> Words["Words:<br/>['i', 'love', 'this', 'amazing', 'product']"]
    
    Words --> Dict{Check Sentiment<br/>Dictionary}
    
    Dict --> Love["'love' → 1.0"]
    Dict --> Amazing["'amazing' → 0.9"]
    Dict --> Other["Other words<br/>not in dictionary"]
    
    Love --> Calc[Calculate Features]
    Amazing --> Calc
    Other --> Calc
    
    Calc --> Features["10 Features:<br/>1. Avg sentiment: 0.95<br/>2. Max positive: 1.0<br/>3. Min sentiment: 0.0<br/>4. Positive count: 0.4<br/>5. Negative count: 0.0<br/>6. Length: 0.25<br/>7. Exclamation: 1.0<br/>8. Question: 0.0<br/>9. Ratio: 0.4<br/>10. Variance: 0.005"]
    
    Features --> Vector["Feature Vector:<br/>[0.95, 1.0, 0.0, 0.4, 0.0,<br/>0.25, 1.0, 0.0, 0.4, 0.005]"]
    
    style Input fill:#e1f5ff
    style Dict fill:#FFD700
    style Features fill:#FFE5B4
    style Vector fill:#90EE90
```

## Class Structure

```mermaid
classDiagram
    class Program {
        +Main(args) void
    }
    
    class SentimentAnalyser {
        -NeuralNetwork network
        -Dictionary~string,double~ wordSentiments
        -int INPUT_SIZE = 10
        -int HIDDEN_SIZE = 8
        -int OUTPUT_SIZE = 3
        +SentimentAnalyser()
        -InitialiseSentimentDictionary() void
        +Train() void
        -ExtractFeatures(sentence) double[]
        +AnalyseSentiment(sentence) string
    }
    
    class NeuralNetwork {
        -int inputSize
        -int hiddenSize
        -int outputSize
        -double[,] weightsInputHidden
        -double[] biasHidden
        -double[,] weightsHiddenOutput
        -double[] biasOutput
        -double[] inputLayer
        -double[] hiddenLayer
        -double[] outputLayer
        -Random random
        +NeuralNetwork(inputSize, hiddenSize, outputSize)
        -Sigmoid(x) double
        -SigmoidDerivative(x) double
        +Forward(input) double[]
        +Backward(expected, learningRate) void
    }
    
    Program --> SentimentAnalyser : creates
    SentimentAnalyser --> NeuralNetwork : contains
```

## Data Flow Summary

```mermaid
sequenceDiagram
    participant U as User
    participant SA as SentimentAnalyser
    participant FE as Feature Extractor
    participant NN as Neural Network
    participant D as Decision Logic
    
    U->>SA: "I love this!"
    SA->>FE: Extract features
    FE->>FE: Tokenize words
    FE->>FE: Look up in dictionary
    FE->>FE: Calculate 10 features
    FE->>SA: [0.95, 1.0, 0.0, ...]
    SA->>NN: Forward pass
    NN->>NN: Input → Hidden (sigmoid)
    NN->>NN: Hidden → Output (sigmoid)
    NN->>SA: [0.92, 0.05, 0.12]
    SA->>D: Find max output
    D->>SA: Index 0 (Positive)
    SA->>U: "Positive"
```

## Legend

- **Dotted lines** (-.->): Many-to-many connections (each input connects to each hidden neuron)
- **Thick lines** (==>): Weighted connections with learned parameters
- **Rectangles**: Processing steps
- **Circles**: Neural network neurons
- **Diamonds**: Decision points
- **Rounded boxes**: Start/End points
