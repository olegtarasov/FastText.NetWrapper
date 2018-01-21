using System;
using System.Runtime.InteropServices;
using System.Text;

namespace FastText.NetWrapper
{
    public partial class FastTextWrapper
    {
        private const string FastTextDll = "FastText.dll";

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct TrainingArgsStruct
        {
            public int Epochs;
            public double LearningRate;
            public int WordNGrams;
            public int MinCharNGrams;
            public int MaxCharNGrams;
            public int Verbose;
        }

        [DllImport(FastTextDll)]
        private static extern IntPtr CreateFastText();

        [DllImport(FastTextDll)]
        private static extern void DestroyFastText(IntPtr hPtr);

        [DllImport(FastTextDll, CharSet = CharSet.Ansi)]
        private static extern void TrainSupervised(IntPtr hPtr, string input, string output, TrainingArgsStruct args);

        [DllImport(FastTextDll, CharSet = CharSet.Ansi)]
        private static extern void LoadModel(IntPtr hPtr, string path);
        
        [DllImport(FastTextDll)]
        private static extern int GetMaxLabelLenght(IntPtr hPtr);
        
        [DllImport(FastTextDll, CharSet = CharSet.Ansi)]
        private static extern float PredictSingle(IntPtr hPtr, string input, StringBuilder predicted);
    }
}