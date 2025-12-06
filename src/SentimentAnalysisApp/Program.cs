using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysisApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create and train the sentiment analyser
            var analyser = new NeuralNetworkSentimentAnalyser();
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
}