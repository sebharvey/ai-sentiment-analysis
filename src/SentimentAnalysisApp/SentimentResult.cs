using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysisApp
{
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
}