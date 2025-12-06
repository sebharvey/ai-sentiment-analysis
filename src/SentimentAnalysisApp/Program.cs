using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TrueNeuralNetworkSentiment
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create and train the sentiment analyser
            var analyser = new TrueNeuralNetworkSentimentAnalyser();
            analyser.Train();

            // Test sentences
            string[] testSentences = new string[]
            {
                "I love this product, it's amazing!",
                "This is terrible and I hate it",
                "It's okay, nothing special",
                "Absolutely brilliant, best purchase ever!",
                "Disappointing and frustrating experience",
                "Pretty good overall",
                "Worst thing I've ever bought",
                "I adore this item", // Word not explicitly in training
                "Fantastic quality"
            };

            Console.WriteLine("True Neural Network Sentiment Analysis\n");
            Console.WriteLine("======================================\n");

            // Run example sentences first
            Console.WriteLine("Example Analyses:\n");
            foreach (var sentence in testSentences)
            {
                var result = analyser.AnalyseSentiment(sentence);
                Console.WriteLine($"Sentence: {sentence}");
                Console.WriteLine($"Sentiment: {result.Sentiment}");
                Console.WriteLine($"Confidence: {result.Confidence:P1}");
                Console.WriteLine($"Scores - Pos: {result.PositiveScore:F3}, " +
                                $"Neg: {result.NegativeScore:F3}, " +
                                $"Neu: {result.NeutralScore:F3}\n");
            }

            // Interactive mode
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("INTERACTIVE MODE - Analyse Your Own Sentences");
            Console.WriteLine(new string('=', 70));
            Console.WriteLine("\nEnter your own sentences to analyse sentiment.");
            Console.WriteLine("Type 'exit' or 'quit' to end the programme.\n");

            while (true)
            {
                Console.Write("Enter sentence: ");
                string userInput = Console.ReadLine();

                // Check for exit commands
                if (string.IsNullOrWhiteSpace(userInput) ||
                    userInput.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                    userInput.Trim().Equals("quit", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("\nThank you for using the sentiment analyser. Goodbye!");
                    break;
                }

                // Analyse the user's sentence
                try
                {
                    var result = analyser.AnalyseSentiment(userInput);
                    Console.WriteLine($"\n  → Sentiment: {result.Sentiment}");
                    Console.WriteLine($"  → Confidence: {result.Confidence:P1}");
                    Console.WriteLine($"  → Scores - Pos: {result.PositiveScore:F3}, " +
                                    $"Neg: {result.NegativeScore:F3}, " +
                                    $"Neu: {result.NeutralScore:F3}\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nError analysing sentence: {ex.Message}\n");
                }
            }
        }
    }

    /// <summary>
    /// Result of sentiment analysis
    /// </summary>
    public class SentimentResult
    {
        public string Sentiment { get; set; }
        public double Confidence { get; set; }
        public double PositiveScore { get; set; }
        public double NegativeScore { get; set; }
        public double NeutralScore { get; set; }
    }

    /// <summary>
    /// A true neural network sentiment analyser that learns word representations
    /// from training data with NO hand-crafted dictionaries
    /// </summary>
    public class TrueNeuralNetworkSentimentAnalyser
    {
        private NeuralNetwork network;
        private Dictionary<string, int> vocabulary;
        private int vocabularySize;
        private const int EMBEDDING_SIZE = 16; // Size of learned word vectors
        private const int HIDDEN_SIZE = 12;
        private const int OUTPUT_SIZE = 3;
        private double[,] wordEmbeddings; // LEARNED word representations

        public TrueNeuralNetworkSentimentAnalyser()
        {
            vocabulary = new Dictionary<string, int>();
        }

        /// <summary>
        /// Builds vocabulary from training data - extracts all unique words
        /// This is the only "pre-processing" - just identifying unique words, not their meanings
        /// </summary>
        private void BuildVocabulary(List<string> sentences)
        {
            var uniqueWords = new HashSet<string>();

            foreach (var sentence in sentences)
            {
                var words = Tokenize(sentence);
                foreach (var word in words)
                {
                    uniqueWords.Add(word);
                }
            }

            // Add special token for unknown words
            vocabulary["<UNK>"] = 0;
            int index = 1;
            foreach (var word in uniqueWords.OrderBy(w => w))
            {
                vocabulary[word] = index++;
            }

            vocabularySize = vocabulary.Count;
            Console.WriteLine($"Built vocabulary of {vocabularySize} words from training data\n");
        }

        /// <summary>
        /// Initializes word embeddings randomly
        /// These will be LEARNED during training to represent word meanings
        /// </summary>
        private void InitializeWordEmbeddings()
        {
            var random = new Random(42);
            wordEmbeddings = new double[vocabularySize, EMBEDDING_SIZE];

            // Initialize with small random values (Xavier initialization)
            double scale = Math.Sqrt(2.0 / (vocabularySize + EMBEDDING_SIZE));
            for (int i = 0; i < vocabularySize; i++)
            {
                for (int j = 0; j < EMBEDDING_SIZE; j++)
                {
                    wordEmbeddings[i, j] = (random.NextDouble() * 2 - 1) * scale;
                }
            }
        }

        /// <summary>
        /// Converts a sentence into a fixed-size feature vector by:
        /// 1. Looking up each word's learned embedding
        /// 2. Averaging all word embeddings in the sentence
        /// This is called "continuous bag of words" representation
        /// </summary>
        private double[] SentenceToVector(string sentence)
        {
            var words = Tokenize(sentence);
            var vector = new double[EMBEDDING_SIZE];

            int wordCount = 0;
            foreach (var word in words)
            {
                int wordIndex;
                if (vocabulary.ContainsKey(word))
                {
                    wordIndex = vocabulary[word];
                }
                else
                {
                    // Unknown word - use the <UNK> token
                    wordIndex = vocabulary["<UNK>"];
                }

                // Add this word's embedding to our sentence vector
                for (int i = 0; i < EMBEDDING_SIZE; i++)
                {
                    vector[i] += wordEmbeddings[wordIndex, i];
                }
                wordCount++;
            }

            // Average the embeddings
            if (wordCount > 0)
            {
                for (int i = 0; i < EMBEDDING_SIZE; i++)
                {
                    vector[i] /= wordCount;
                }
            }

            return vector;
        }

        /// <summary>
        /// Updates word embeddings based on the gradient
        /// This is how the network LEARNS what words mean!
        /// </summary>
        private void UpdateWordEmbeddings(string sentence, double[] embeddingGradient, double learningRate)
        {
            var words = Tokenize(sentence);
            int wordCount = words.Length;

            if (wordCount == 0) return;

            // Distribute the gradient back to each word's embedding
            foreach (var word in words)
            {
                int wordIndex;
                if (vocabulary.ContainsKey(word))
                {
                    wordIndex = vocabulary[word];
                }
                else
                {
                    wordIndex = vocabulary["<UNK>"];
                }

                // Update this word's embedding
                for (int i = 0; i < EMBEDDING_SIZE; i++)
                {
                    wordEmbeddings[wordIndex, i] += learningRate * embeddingGradient[i] / wordCount;
                }
            }
        }

        /// <summary>
        /// Simple tokenization - splits on whitespace and punctuation
        /// </summary>
        private string[] Tokenize(string sentence)
        {
            return sentence.ToLower()
                          .Split(new[] { ' ', ',', '.', '!', '?', ';', ':', '"', '\'' },
                                StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>
        /// Loads training data from a CSV file
        /// Format: sentiment,sentence
        /// Where sentiment is: positive, negative, or neutral
        /// </summary>
        private List<(string sentence, double[] label)> LoadTrainingData(string filePath)
        {
            var trainingData = new List<(string sentence, double[] label)>();

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Training data file not found: {filePath}");
            }

            var lines = File.ReadAllLines(filePath);
            int lineNumber = 0;

            foreach (var line in lines)
            {
                lineNumber++;

                // Skip empty lines or comments
                if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("#"))
                    continue;

                var parts = line.Split(',', 2); // Split on first comma only
                if (parts.Length != 2)
                {
                    Console.WriteLine($"Warning: Skipping malformed line {lineNumber}: {line}");
                    continue;
                }

                var sentiment = parts[0].Trim().ToLower();
                var sentence = parts[1].Trim();

                double[] label = sentiment switch
                {
                    "positive" => new double[] { 1, 0, 0 },
                    "negative" => new double[] { 0, 1, 0 },
                    "neutral" => new double[] { 0, 0, 1 },
                    _ => null
                };

                if (label == null)
                {
                    Console.WriteLine($"Warning: Unknown sentiment '{sentiment}' on line {lineNumber}, skipping");
                    continue;
                }

                trainingData.Add((sentence, label));
            }

            Console.WriteLine($"Loaded {trainingData.Count} training examples from {filePath}\n");
            return trainingData;
        }

        /// <summary>
        /// Trains the network on labelled sentiment data
        /// The network learns BOTH word embeddings AND classification weights
        /// </summary>
        public void Train()
        {
            // Load training data from file
            var trainingData = LoadTrainingData("training_data.csv");

            Console.WriteLine("Training True Neural Network for Sentiment Analysis");
            Console.WriteLine("===================================================\n");

            // Step 1: Build vocabulary from training sentences
            BuildVocabulary(trainingData.Select(t => t.sentence).ToList());

            // Step 2: Initialize word embeddings randomly
            InitializeWordEmbeddings();

            // Step 3: Create the neural network
            network = new NeuralNetwork(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

            // Step 4: Train the network
            int epochs = 2000;
            double learningRate = 0.05;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;

                // Shuffle training data each epoch for better learning
                var shuffled = trainingData.OrderBy(x => Guid.NewGuid()).ToList();

                foreach (var (sentence, label) in shuffled)
                {
                    // Convert sentence to vector using current word embeddings
                    double[] sentenceVector = SentenceToVector(sentence);

                    // Forward pass through network
                    double[] output = network.Forward(sentenceVector);

                    // Calculate error
                    double error = 0;
                    for (int i = 0; i < output.Length; i++)
                    {
                        error += Math.Pow(label[i] - output[i], 2);
                    }
                    totalError += error;

                    // Backward pass through network
                    double[] embeddingGradient = network.Backward(label, learningRate);

                    // Update word embeddings (this is where words LEARN their meanings!)
                    UpdateWordEmbeddings(sentence, embeddingGradient, learningRate);
                }

                if ((epoch + 1) % 200 == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError:F4}");
                }
            }

            Console.WriteLine("\nTraining complete!\n");
            Console.WriteLine("The network has learned:");
            Console.WriteLine("1. Word embeddings (what each word means)");
            Console.WriteLine("2. How to classify sentiments from word meanings\n");
        }

        /// <summary>
        /// Analyses sentiment of any sentence, even with words not seen during training
        /// </summary>
        public SentimentResult AnalyseSentiment(string sentence)
        {
            // Convert sentence to vector using learned embeddings
            double[] sentenceVector = SentenceToVector(sentence);

            // Get prediction from network
            double[] output = network.Forward(sentenceVector);

            // Find the highest scoring sentiment
            int maxIndex = 0;
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > output[maxIndex])
                {
                    maxIndex = i;
                }
            }

            string[] sentiments = { "Positive", "Negative", "Neutral" };

            return new SentimentResult
            {
                Sentiment = sentiments[maxIndex],
                Confidence = output[maxIndex],
                PositiveScore = output[0],
                NegativeScore = output[1],
                NeutralScore = output[2]
            };
        }
    }

    /// <summary>
    /// Neural network that can return gradients for embedding updates
    /// </summary>
    public class NeuralNetwork
    {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;

        private double[,] weightsInputHidden;
        private double[] biasHidden;
        private double[,] weightsHiddenOutput;
        private double[] biasOutput;

        private double[] inputLayer;
        private double[] hiddenLayer;
        private double[] outputLayer;

        private Random random;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            random = new Random(42);

            // Initialize weights
            weightsInputHidden = new double[inputSize, hiddenSize];
            biasHidden = new double[hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            biasOutput = new double[outputSize];

            // Xavier initialization for better training
            double scaleInputHidden = Math.Sqrt(2.0 / (inputSize + hiddenSize));
            double scaleHiddenOutput = Math.Sqrt(2.0 / (hiddenSize + outputSize));

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = (random.NextDouble() * 2 - 1) * scaleInputHidden;

            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = (random.NextDouble() * 2 - 1) * scaleHiddenOutput;

            for (int i = 0; i < hiddenSize; i++)
                biasHidden[i] = 0.01;

            for (int i = 0; i < outputSize; i++)
                biasOutput[i] = 0.01;
        }

        private double Sigmoid(double x)
        {
            // Clip to prevent overflow
            if (x > 20) return 1.0;
            if (x < -20) return 0.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }

        public double[] Forward(double[] input)
        {
            inputLayer = input;
            hiddenLayer = new double[hiddenSize];
            outputLayer = new double[outputSize];

            // Input to hidden
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputHidden[i, j];
                }
                hiddenLayer[j] = Sigmoid(sum);
            }

            // Hidden to output
            for (int k = 0; k < outputSize; k++)
            {
                double sum = biasOutput[k];
                for (int j = 0; j < hiddenSize; j++)
                {
                    sum += hiddenLayer[j] * weightsHiddenOutput[j, k];
                }
                outputLayer[k] = Sigmoid(sum);
            }

            return outputLayer;
        }

        /// <summary>
        /// Backward pass that returns gradient for the input (embeddings)
        /// This allows word embeddings to be updated based on classification error
        /// </summary>
        public double[] Backward(double[] expected, double learningRate)
        {
            // Output layer deltas
            double[] outputDelta = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double error = expected[k] - outputLayer[k];
                outputDelta[k] = error * SigmoidDerivative(outputLayer[k]);
            }

            // Hidden layer deltas
            double[] hiddenDelta = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double error = 0;
                for (int k = 0; k < outputSize; k++)
                {
                    error += outputDelta[k] * weightsHiddenOutput[j, k];
                }
                hiddenDelta[j] = error * SigmoidDerivative(hiddenLayer[j]);
            }

            // Calculate gradient for input (this will update word embeddings!)
            double[] inputGradient = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                double gradient = 0;
                for (int j = 0; j < hiddenSize; j++)
                {
                    gradient += hiddenDelta[j] * weightsInputHidden[i, j];
                }
                inputGradient[i] = gradient;
            }

            // Update hidden to output weights
            for (int j = 0; j < hiddenSize; j++)
            {
                for (int k = 0; k < outputSize; k++)
                {
                    weightsHiddenOutput[j, k] += learningRate * outputDelta[k] * hiddenLayer[j];
                }
            }
            for (int k = 0; k < outputSize; k++)
            {
                biasOutput[k] += learningRate * outputDelta[k];
            }

            // Update input to hidden weights
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] += learningRate * hiddenDelta[j] * inputLayer[i];
                }
            }
            for (int j = 0; j < hiddenSize; j++)
            {
                biasHidden[j] += learningRate * hiddenDelta[j];
            }

            return inputGradient; // Return gradient to update word embeddings
        }
    }
}