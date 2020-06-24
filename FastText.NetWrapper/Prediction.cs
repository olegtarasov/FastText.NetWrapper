using System.Diagnostics;

namespace FastText.NetWrapper
{
    /// <summary>
    /// Represents a single label prediction.
    /// </summary>
    [DebuggerDisplay("{Label} ({Probability})")]
    public struct Prediction
    {
        /// <summary>
        /// Label probability.
        /// </summary>
        public readonly float Probability;

        /// <summary>
        /// Label text.
        /// </summary>
        public readonly string Label;

        /// <summary>
        /// Ctor.
        /// </summary>
        public Prediction(float probability, string label)
        {
            Probability = probability;
            Label = label;
        }
    }
}