using System;
using System.Runtime.InteropServices;
using System.Text;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper
    {
        internal const string FastTextDll = "fasttext";

        internal enum model_name : int { cbow = 1, sg, sup };

        internal enum loss_name : int { hs = 1, ns, softmax, ova };

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct SupervisedArgsStruct
        {
            public int Epochs;
            public double LearningRate;
            public int WordNGrams;
            public int MinCharNGrams;
            public int MaxCharNGrams;
            public int Verbose;
            public int Threads;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct FastTextArgsStruct
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
            public bool saveOutput;
            public int seed;
            public bool qout;
            public bool retrain;
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
        private static extern int Supervised(IntPtr hPtr, string input, string output, FastTextArgsStruct trainArgs, string labelPrefix, string pretrainedVectors);
        
        [DllImport(FastTextDll)]
        private static extern int GetNN(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);

        [DllImport(FastTextDll)]
        private static extern int GetSentenceVector(IntPtr hPtr, byte[] input, IntPtr vector);
        
        #endregion

        #region Predictions

        [DllImport(FastTextDll)]
        private static extern float PredictSingle(IntPtr hPtr, byte[] input, IntPtr predicted);

        [DllImport(FastTextDll)]
        private static extern int PredictMultiple(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);

        #endregion

        #region Testing
        
        [DllImport(FastTextDll)]
        private static extern int Test(IntPtr hPtr, string input, int k, float threshold, IntPtr meterPtr);
        
        [DllImport(FastTextDll)]
        private static extern int DestroyMeter(IntPtr hPtr);

        #endregion
        
        #region Deprecated
        
        [DllImport(FastTextDll)]
        private static extern int TrainSupervised(IntPtr hPtr, string input, string output, SupervisedArgsStruct args, string labelPrefix);

        [DllImport(FastTextDll)]
        private static extern int Train(IntPtr hPtr, string input, string output, FastTextArgsStruct args, string labelPrefix, string pretrainedVectors);
        
        #endregion
    }
}