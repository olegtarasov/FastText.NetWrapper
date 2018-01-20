using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace FastText.NetWrapper
{
    public class FastTextWrapper : IDisposable
    {
        private const string FastTextDll = "FastText.dll";

        private IntPtr _fastText;

        public FastTextWrapper()
        {
            _fastText = CreateFastText();
        }

        public void Train(string inputPath, string outputPath)
        {
            TrainSupervised(_fastText, inputPath, outputPath);
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

        [DllImport(FastTextDll)]
        private static extern IntPtr CreateFastText();

        [DllImport(FastTextDll)]
        private static extern void DestroyFastText(IntPtr hPtr);

        [DllImport(FastTextDll, CharSet = CharSet.Ansi)]
        private static extern void TrainSupervised(IntPtr hPtr, string input, string output);
    }
}
