using System;
using System.Runtime.InteropServices;
using System.Text;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper
    {
        private const string FastTextDll = "fasttext";

        private enum model_name : int { cbow = 1, sg, sup };
        private enum loss_name : int { hs = 1, ns, softmax, ova };

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
        private struct TrainingArgsStruct
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
            public bool qout;
            public bool retrain;
            public bool qnorm;
            public ulong cutoff;
            public ulong dsub;
        }

        [DllImport(FastTextDll)]
        private static extern IntPtr CreateFastText();

        [DllImport(FastTextDll)]
        private static extern void DestroyFastText(IntPtr hPtr);

        [DllImport(FastTextDll)]
        private static extern void DestroyString(IntPtr str);

        [DllImport(FastTextDll)]
        private static extern void DestroyStrings(IntPtr strings, int cnt);

        [DllImport(FastTextDll)]
        private static extern void TrainSupervised(IntPtr hPtr, string input, string output, SupervisedArgsStruct args, string labelPrefix);

        [DllImport(FastTextDll)]
        private static extern void Train(IntPtr hPtr, string input, string output, TrainingArgsStruct args, string labelPrefix, string pretrainedVectors);

        [DllImport(FastTextDll)]
        private static extern void LoadModel(IntPtr hPtr, string path);
        
        [DllImport(FastTextDll)]
        private static extern int GetMaxLabelLength(IntPtr hPtr);

        [DllImport(FastTextDll)]
        private static extern int GetLabels(IntPtr hPtr, IntPtr labels);

        [DllImport(FastTextDll)]
        private static extern int GetNN(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);
        
        [DllImport(FastTextDll)]
        private static extern float PredictSingle(IntPtr hPtr, byte[] input, IntPtr predicted);

        [DllImport(FastTextDll)]
        private static extern int PredictMultiple(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);

        [DllImport(FastTextDll)]
        private static extern  int GetSentenceVector(IntPtr hPtr, byte[] input, IntPtr vector);

        [DllImport(FastTextDll)]
        private static  extern  void DestroyVector(IntPtr vector);
    }
}