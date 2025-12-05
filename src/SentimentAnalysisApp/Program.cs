using System;
using System.Collections.Generic;
using System.Linq;

namespace SentimentAnalysisApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create and train the sentiment analyser
            var analyser = new SentimentAnalyser();
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
                "Worst thing I've ever bought"
            };

            Console.WriteLine("Sentiment Analysis Neural Network\n");
            Console.WriteLine("=================================\n");

            foreach (var sentence in testSentences)
            {
                string sentiment = analyser.AnalyseSentiment(sentence);
                Console.WriteLine($"Sentence: {sentence}");
                Console.WriteLine($"Sentiment: {sentiment}\n");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }

    public class SentimentAnalyser
    {
        private NeuralNetwork network;
        private Dictionary<string, double> wordSentiments;
        private const int INPUT_SIZE = 10; // Number of features we'll extract
        private const int HIDDEN_SIZE = 8;  // Hidden layer neurons
        private const int OUTPUT_SIZE = 3;  // Positive, Negative, Neutral

        public SentimentAnalyser()
        {
            // Initialise the neural network with our layer sizes
            network = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            
            // Create a simple sentiment dictionary for feature extraction
            InitialiseSentimentDictionary();
        }

        /// <summary>
        /// Initialises a dictionary of words with sentiment scores
        /// Positive words have positive scores, negative words have negative scores
        /// </summary>
        private void InitialiseSentimentDictionary()
        {
            wordSentiments = new Dictionary<string, double>
            {
                // Positive words
                { "love", 1.0 }, { "great", 0.8 }, { "excellent", 0.9 }, { "amazing", 0.9 },
                { "wonderful", 0.8 }, { "fantastic", 0.9 }, { "good", 0.6 }, { "brilliant", 0.9 },
                { "best", 0.9 }, { "happy", 0.7 }, { "perfect", 0.9 }, { "beautiful", 0.8 },
                
                // Negative words
                { "hate", -1.0 }, { "terrible", -0.9 }, { "awful", -0.9 }, { "bad", -0.7 },
                { "horrible", -0.9 }, { "worst", -1.0 }, { "poor", -0.6 }, { "disappointing", -0.7 },
                { "frustrating", -0.7 }, { "annoying", -0.6 }, { "useless", -0.8 },
                
                // Neutral/mild words
                { "okay", 0.1 }, { "fine", 0.2 }, { "acceptable", 0.2 }, { "average", 0.0 }
            };
        }

        /// <summary>
        /// Trains the neural network with sample data
        /// In a real application, you'd use a large labelled dataset
        /// </summary>
        public void Train()
        {
            // Training data: pairs of sentences and their expected outputs
            // Output format: [positive_score, negative_score, neutral_score]
            var trainingData = new List<(string sentence, double[] expected)>
            {
                // Positive examples
                ("I love this it's amazing", new double[] { 1, 0, 0 }),
                ("Great product excellent quality", new double[] { 1, 0, 0 }),
                ("Wonderful experience very happy", new double[] { 1, 0, 0 }),
                ("Fantastic brilliant and beautiful", new double[] { 1, 0, 0 }),
                ("Best thing ever so good", new double[] { 1, 0, 0 }),
                
                // Negative examples
                ("I hate this terrible product", new double[] { 0, 1, 0 }),
                ("Awful experience very bad", new double[] { 0, 1, 0 }),
                ("Horrible and disappointing worst ever", new double[] { 0, 1, 0 }),
                ("Frustrating and useless", new double[] { 0, 1, 0 }),
                ("Poor quality hate it", new double[] { 0, 1, 0 }),
                
                // Neutral examples
                ("It's okay nothing special", new double[] { 0, 0, 1 }),
                ("Average product acceptable", new double[] { 0, 0, 1 }),
                ("Fine I suppose", new double[] { 0, 0, 1 }),
                ("Okay quality average experience", new double[] { 0, 0, 1 })
            };

            // Train for multiple epochs (passes through the data)
            int epochs = 1000;
            double learningRate = 0.1;

            Console.WriteLine("Training neural network...\n");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;

                // Go through each training example
                foreach (var (sentence, expected) in trainingData)
                {
                    // Extract features from the sentence
                    double[] features = ExtractFeatures(sentence);

                    // Forward pass: get the network's prediction
                    double[] output = network.Forward(features);

                    // Calculate error
                    double error = 0;
                    for (int i = 0; i < output.Length; i++)
                    {
                        error += Math.Pow(expected[i] - output[i], 2);
                    }
                    totalError += error;

                    // Backward pass: update weights based on error
                    network.Backward(expected, learningRate);
                }

                // Print progress every 100 epochs
                if ((epoch + 1) % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {totalError:F4}");
                }
            }

            Console.WriteLine("\nTraining complete!\n");
        }

        /// <summary>
        /// Extrazes features from a sentence to feed into the neural network
        /// Returns a fixed-size array of numerical features
        /// </summary>
        private double[] ExtractFeatures(string sentence)
        {
            sentence = sentence.ToLower();
            string[] words = sentence.Split(new[] { ' ', ',', '.', '!', '?' }, 
                                          StringSplitOptions.RemoveEmptyEntries);

            double[] features = new double[INPUT_SIZE];

            // Feature 1: Average sentiment score from dictionary
            double sentimentSum = 0;
            int sentimentWordCount = 0;
            foreach (var word in words)
            {
                if (wordSentiments.ContainsKey(word))
                {
                    sentimentSum += wordSentiments[word];
                    sentimentWordCount++;
                }
            }
            features[0] = sentimentWordCount > 0 ? sentimentSum / sentimentWordCount : 0;

            // Feature 2: Maximum positive sentiment in sentence
            features[1] = words.Where(w => wordSentiments.ContainsKey(w))
                              .Select(w => wordSentiments[w])
                              .DefaultIfEmpty(0)
                              .Max();

            // Feature 3: Minimum sentiment (most negative) in sentence
            features[2] = words.Where(w => wordSentiments.ContainsKey(w))
                              .Select(w => wordSentiments[w])
                              .DefaultIfEmpty(0)
                              .Min();

            // Feature 4: Count of positive words
            features[3] = words.Count(w => wordSentiments.ContainsKey(w) && wordSentiments[w] > 0.5) / 10.0;

            // Feature 5: Count of negative words
            features[4] = words.Count(w => wordSentiments.ContainsKey(w) && wordSentiments[w] < -0.5) / 10.0;

            // Feature 6: Sentence length (normalized)
            features[5] = Math.Min(words.Length / 20.0, 1.0);

            // Feature 7: Presence of exclamation marks
            features[6] = sentence.Contains('!') ? 1.0 : 0.0;

            // Feature 8: Presence of question marks
            features[7] = sentence.Contains('?') ? 1.0 : 0.0;

            // Feature 9: Ratio of sentiment words to total words
            features[8] = words.Length > 0 ? (double)sentimentWordCount / words.Length : 0;

            // Feature 10: Variance in sentiment scores
            if (sentimentWordCount > 1)
            {
                var sentiments = words.Where(w => wordSentiments.ContainsKey(w))
                                     .Select(w => wordSentiments[w]).ToList();
                double mean = sentiments.Average();
                double variance = sentiments.Sum(s => Math.Pow(s - mean, 2)) / sentiments.Count;
                features[9] = variance;
            }
            else
            {
                features[9] = 0;
            }

            return features;
        }

        /// <summary>
        /// Analyses the sentiment of a given sentence
        /// Returns "Positive", "Negative", or "Neutral"
        /// </summary>
        public string AnalyseSentiment(string sentence)
        {
            // Extract features from the sentence
            double[] features = ExtractFeatures(sentence);

            // Get prediction from neural network
            double[] output = network.Forward(features);

            // Find which output neuron has the highest activation
            int maxIndex = 0;
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > output[maxIndex])
                {
                    maxIndex = i;
                }
            }

            // Map the output to sentiment labels
            string[] sentiments = { "Positive", "Negative", "Neutral" };
            return sentiments[maxIndex];
        }
    }

    /// <summary>
    /// A simple feedforward neural network with one hidden layer
    /// Uses sigmoid activation function and backpropagation for training
    /// </summary>
    public class NeuralNetwork
    {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;

        // Weights and biases for input->hidden layer
        private double[,] weightsInputHidden;
        private double[] biasHidden;

        // Weights and biases for hidden->output layer
        private double[,] weightsHiddenOutput;
        private double[] biasOutput;

        // Store activations for backpropagation
        private double[] inputLayer;
        private double[] hiddenLayer;
        private double[] outputLayer;

        private Random random;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            random = new Random(42); // Fixed seed for reproducibility

            // Initialise weights with small random values
            weightsInputHidden = new double[inputSize, hiddenSize];
            biasHidden = new double[hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            biasOutput = new double[outputSize];

            // Initialise weights randomly between -0.5 and 0.5
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = random.NextDouble() - 0.5;

            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = random.NextDouble() - 0.5;

            // Initialise biases to small random values
            for (int i = 0; i < hiddenSize; i++)
                biasHidden[i] = random.NextDouble() - 0.5;

            for (int i = 0; i < outputSize; i++)
                biasOutput[i] = random.NextDouble() - 0.5;
        }

        /// <summary>
        /// Sigmoid activation function: maps any value to range (0, 1)
        /// </summary>
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Derivative of sigmoid function, used in backpropagation
        /// </summary>
        private double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }

        /// <summary>
        /// Forward pass: propagate input through the network to get output
        /// </summary>
        public double[] Forward(double[] input)
        {
            inputLayer = input;
            hiddenLayer = new double[hiddenSize];
            outputLayer = new double[outputSize];

            // Calculate hidden layer activations
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputHidden[i, j];
                }
                hiddenLayer[j] = Sigmoid(sum);
            }

            // Calculate output layer activations
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
        /// Backward pass: update weights based on error using gradient descent
        /// </summary>
        public void Backward(double[] expected, double learningRate)
        {
            // Calculate output layer error and delta
            double[] outputDelta = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double error = expected[k] - outputLayer[k];
                outputDelta[k] = error * SigmoidDerivative(outputLayer[k]);
            }

            // Calculate hidden layer error and delta
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

            // Update weights and biases for hidden->output layer
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

            // Update weights and biases for input->hidden layer
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
        }
    }
}