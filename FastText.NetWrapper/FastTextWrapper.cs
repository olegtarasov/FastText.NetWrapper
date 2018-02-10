using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace FastText.NetWrapper
{
    /// <summary>
    /// A wrapper around native fastText implementation.
    /// </summary>
    public partial class FastTextWrapper : IDisposable
    {
        private static readonly ILog _log = LogManager.GetLogger(typeof(FastTextWrapper));
        private static readonly Encoding _utf8 = Encoding.UTF8;
        
        private IntPtr _fastText;
        private int _maxLabelLen;

        /// <summary>
        /// Ctor.
        /// </summary>
        public FastTextWrapper()
        {
            LoadNativeLibrary();
            _fastText = CreateFastText();
        }

        /// <summary>
        /// Loads a trained model from a file.
        /// </summary>
        /// <param name="path">Path to a model (.bin file).</param>
        public void LoadModel(string path)
        {
            LoadModel(_fastText, path);
            _maxLabelLen = GetMaxLabelLenght(_fastText);
        }

        /// <summary>
        /// Predicts a single label from input text.
        /// </summary>
        /// <param name="text">Text to predict a label from.</param>
        /// <returns>Single prediction.</returns>
        public Prediction PredictSingle(string text)
        {
            if (_maxLabelLen == 0)
            {
                throw new InvalidOperationException("Model not loaded!");
            }

            var builder = new StringBuilder(_maxLabelLen + 1);
            float intensity = PredictSingle(_fastText, _utf8.GetBytes(text), builder);
            float prob = Math.Exp(intensity);
            
            return new Prediction(prob, builder.ToString());
        }

        /// <summary>
        /// Trains a new model.
        /// </summary>
        /// <param name="inputPath">Path to a training set.</param>
        /// <param name="outputPath">Path to write the model to (excluding extension).</param>
        /// <param name="args">Training arguments.</param>
        /// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
        public void Train(string inputPath, string outputPath, TrainingArgs args)
        {
            var argsStruct = new TrainingArgsStruct
                             {
                                 Epochs = args.Epochs,
                                 LearningRate = args.LearningRate,
                                 MaxCharNGrams = args.MaxCharNGrams,
                                 MinCharNGrams = args.MinCharNGrams,
                                 Verbose = args.Verbose,
                                 WordNGrams = args.WordNGrams
                             };
            
            TrainSupervised(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix);
            _maxLabelLen = GetMaxLabelLenght(_fastText);
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (_fastText == IntPtr.Zero)
            {
                return;
            }

            DestroyFastText(_fastText);
            _fastText = IntPtr.Zero;
        }
    }
}
