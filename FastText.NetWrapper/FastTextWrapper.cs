using System;
using System.Runtime.InteropServices;
using System.Text;
using FastText.NetWrapper.Logging;

namespace FastText.NetWrapper
{
    /// <summary>
    /// A wrapper around native fastText implementation.
    /// </summary>
    public partial class FastTextWrapper : IDisposable
    {
        private static readonly ILog _log = LogProvider.For<FastTextWrapper>();
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
        /// Gets all labels that classifier was trained on.
        /// </summary>
        /// <returns>Labels.</returns>
        public unsafe string[] GetLabels()
        {
            IntPtr labelsPtr;
            int numLabels = GetLabels(_fastText, new IntPtr(&labelsPtr));

            var result = new string[numLabels];
            for (int i = 0; i < numLabels; i++)
            {
                var ptr = Marshal.ReadIntPtr(labelsPtr, i * IntPtr.Size);
                // TODO: Support UTF-8 labels
                result[i] = Marshal.PtrToStringAnsi(ptr);
            }

            DestroyStrings(labelsPtr, numLabels);

            return result;
        }

        /// <summary>
        /// Predicts a single label from input text.
        /// </summary>
        /// <param name="text">Text to predict a label from.</param>
        /// <returns>Single prediction.</returns>
        public unsafe Prediction PredictSingle(string text)
        {
            if (_maxLabelLen == 0)
            {
                throw new InvalidOperationException("Model not loaded!");
            }

            IntPtr labelPtr;
            float prob = PredictSingle(_fastText, _utf8.GetBytes(text), new IntPtr(&labelPtr));
            
            // TODO: We are assuming ASCII strings, but they are UTF-8
            string label = Marshal.PtrToStringAnsi(labelPtr);
            DestroyString(labelPtr);

            return new Prediction(prob, label);
        }

        /// <summary>
        /// Predicts multiple labels from input text.
        /// </summary>
        /// <param name="text">Text to predict labels from.</param>
        /// <param name="number">Number of labels to predict.</param>
        /// <returns>Multiple predictions.</returns>
        public unsafe Prediction[] PredictMultiple(string text, int number)
        {
            if (_maxLabelLen == 0)
            {
                throw new InvalidOperationException("Model not loaded!");
            }

            var probs = new float[number];
            IntPtr labelsPtr;
            
            int cnt = PredictMultiple(_fastText, _utf8.GetBytes(text), new IntPtr(&labelsPtr), probs, number);
            var result = new Prediction[cnt];

            for (int i = 0; i < cnt; i++)
            {
                var ptr = Marshal.ReadIntPtr(labelsPtr, i * IntPtr.Size);
                // TODO: Support UTF-8 labels
                string label = Marshal.PtrToStringAnsi(ptr);
                result[i] = new Prediction(probs[i], label);
            }

            DestroyStrings(labelsPtr, cnt);

            return result;
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
