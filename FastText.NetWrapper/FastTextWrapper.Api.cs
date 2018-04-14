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

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr CreateFastText();

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DestroyFastText(IntPtr hPtr);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DestroyString(IntPtr str);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DestroyStrings(IntPtr strings, int cnt);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern void TrainSupervised(IntPtr hPtr, string input, string output, TrainingArgsStruct args, string labelPrefix);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern void LoadModel(IntPtr hPtr, string path);
        
        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int GetMaxLabelLenght(IntPtr hPtr);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int GetLabels(IntPtr hPtr, IntPtr labels);
        
        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern float PredictSingle(IntPtr hPtr, byte[] input, IntPtr predicted);

        [DllImport(FastTextDll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern int PredictMultiple(IntPtr hPtr, byte[] input, IntPtr predictedLabels, float[] predictedProbabilities, int n);
    }
}