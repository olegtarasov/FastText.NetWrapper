using System;

namespace FastText.NetWrapper
{
    public class NativeLibraryException : Exception
    {
        public NativeLibraryException()
        {
        }

        public NativeLibraryException(string message) : base(message)
        {
        }
    }
}