using System;

namespace FastText.NetWrapper
{
    /// <summary>
    /// FastText training arguments.
    /// </summary>
    public class TrainingArgs
    {
        /// <summary>
        /// Number of epochs to train, standard range [5 - 50].
        /// </summary>
        public int Epochs { get; set; } = 5;

        /// <summary>
        /// Learning rate, standard range [0.1 - 1.0].
        /// </summary>
        public double LearningRate { get; set; } = 0.05;

        /// <summary>
        /// Use word n-grams, standard range [1 - 5].
        /// </summary>
        public int WordNGrams { get; set; } = 1;

        /// <summary>
        /// Use char-level n-grams, min length.
        /// </summary>
        public int MinCharNGrams { get; set; } = 3;
        
        /// <summary>
        /// Use char-level n-grams, max length.
        /// </summary>
        public int MaxCharNGrams { get; set; } = 6;

        /// <summary>
        /// Verbosity level [0..2].
        /// </summary>
        public int Verbose { get; set; } = 0;

        /// <summary>
        /// Labels prefix [__label__].
        /// </summary>
        public string LabelPrefix { get; set; }

        /// <summary>
        /// Creates a new instance with default arguments for supervised training.
        /// </summary>
        /// <param name="builder">An optional action to change some params.</param>
        public static TrainingArgs SupervisedDefaults(Action<TrainingArgs> builder = null)
        {
            var result = new TrainingArgs
            {
                LearningRate = 0.1,
                MaxCharNGrams = 0,
                MinCharNGrams = 0
            };

            builder?.Invoke(result);

            return result;
        }
    }
}