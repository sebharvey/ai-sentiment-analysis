using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysisApp
{
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