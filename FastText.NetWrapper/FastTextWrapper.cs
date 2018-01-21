using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper : IDisposable
    {
        private static readonly ILog _log = LogManager.GetLogger(typeof(FastTextWrapper));
        
        private IntPtr _fastText;
        private int _maxLabelLen;

        public FastTextWrapper()
        {
            LoadNativeLibrary();
            _fastText = CreateFastText();
        }

        public void LoadModel(string path)
        {
            LoadModel(_fastText, path);
            _maxLabelLen = GetMaxLabelLenght(_fastText);
        }

        public Prediction PredictSingle(string text)
        {
            if (_maxLabelLen == 0)
            {
                throw new InvalidOperationException("Model not loaded!");
            }

            var builder = new StringBuilder(_maxLabelLen + 1);
            float prob = PredictSingle(_fastText, text, builder);

            return new Prediction(prob, builder.ToString());
        }

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
            
            TrainSupervised(_fastText, inputPath, outputPath, argsStruct);
            _maxLabelLen = GetMaxLabelLenght(_fastText);
        }

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
