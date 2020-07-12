using System;
using System.Runtime.InteropServices;
using System.Text;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper
    {
        internal const string FastTextDll = "fasttext";

        public enum model_name : int { cbow = 1, sg, sup };

        public enum loss_name : int { hs = 1, ns, softmax, ova };

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct AutotuneArgsStruct
        {
            public string ValidationFile;
            public string Metric;
            public int Predictions;
            public int Duration;
            public string ModelSize;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct FastTextArgsStruct
        {
            public double lr;
            public int lrUpdateRate;
            public int dim;
            public int ws;
            public int epoch;
            public int minCount;
            public int minCountLabel;
            public int neg;
            public int wordNgrams;
            public loss_name loss;
            public model_name model;
            public int bucket;
            public int minn;
            public int maxn;
            public int thread;
            public double t;
            public int verbose;
            
            [MarshalAs(UnmanagedType.I1)]
            public bool saveOutput;
            
            public int seed;
            
            [MarshalAs(UnmanagedType.I1)]
            public bool qout;
            
            [MarshalAs(UnmanagedType.I1)]
            public bool retrain;
            
            [MarshalAs(UnmanagedType.I1)]
            public bool qnorm;
            
            public ulong cutoff;
            public ulong dsub;
        }
        
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct TestMetrics
        {
            public long Gold;
            public long Predicted;
            public long PredictedGold;
            public int ScoresLen;
            public int Label;
            public IntPtr PredictedScores;
            public IntPtr GoldScores;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct TestMeter
        {
            public long Examples;
            public long Labels;
            private IntPtr SourceMeter;
            public IntPtr Metrics;
            public IntPtr LabelMetrics;
        }

        #region Errors

        [DllImport(FastTextDll)]
        private static extern void GetLastErrorText(IntPtr error);

        #endregion
        
        #region Model management
        
        [DllImport(FastTextDll)]
        private static extern IntPtr CreateFastText();
        
        [DllImport(FastTextDll)]
        private static extern int LoadModel(IntPtr hPtr, string path);

        [DllImport(FastTextDll)]
        private static extern int LoadModelData(IntPtr hPtr, byte[] data, long length);

        [DllImport(FastTextDll)]
        private static extern void DestroyFastText(IntPtr hPtr);
        
        #endregion
        
        #region Resource management
        
        [DllImport(FastTextDll)]
        private static extern void DestroyString(IntPtr str);

        [DllImport(FastTextDll)]
        private static extern void DestroyStrings(IntPtr strings, int cnt);

        [DllImport(FastTextDll)]
        private static  extern  void DestroyVector(IntPtr vector);
        
        #endregion

        #region Label info

        [DllImport(FastTextDll)]
        private static extern int GetMaxLabelLength(IntPtr hPtr);

        [DllImport(FastTextDll)]
        private static extern int GetLabels(IntPtr hPtr, IntPtr labels);

        [DllImport(FastTextDll)]
        private static extern bool IsModelReady(IntPtr hPtr);
        
        [DllImport(FastTextDll)]
        private static extern int GetModelDimension(IntPtr hPtr);

        #endregion

        #region FastText commands

        [DllImport(FastTextDll)]
        private static extern int Train(IntPtr hPtr, string input, string output, FastTextArgsStruct trainArgs, AutotuneArgsStruct tuneArgs, string labelPrefix, string pretrainedVectors, [MarshalAs(UnmanagedType.I1)] bool debug);
        
        [DllImport(FastTextDll)]
        private static extern int Quantize(IntPtr hPtr, string output, FastTextArgsStruct trainArgs, string label);
        
        [DllImport(FastTextDll)]
        private static extern int GetNN(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);

        [DllImport(FastTextDll)]
        private static extern int GetSentenceVector(IntPtr hPtr, byte[] input, IntPtr vector);

        [DllImport(FastTextDll)]
        private static extern int GetWordVector(IntPtr hPtr, byte[] word, IntPtr vector);
        
        #endregion

        #region Predictions

        [DllImport(FastTextDll)]
        private static extern float PredictSingle(IntPtr hPtr, byte[] input, IntPtr predicted);

        [DllImport(FastTextDll)]
        private static extern int PredictMultiple(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);

        #endregion

        #region Testing
        
        [DllImport(FastTextDll)]
        private static extern int Test(IntPtr hPtr, string input, int k, float threshold, IntPtr meterPtr, [MarshalAs(UnmanagedType.I1)] bool debug);
        
        [DllImport(FastTextDll)]
        private static extern int DestroyMeter(IntPtr hPtr);

        #endregion
    }
}