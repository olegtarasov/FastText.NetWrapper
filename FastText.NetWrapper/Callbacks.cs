using System.Runtime.InteropServices;

namespace FastText.NetWrapper
{
    /// <summary>
    /// Training progress callback.
    /// </summary>
    /// <param name="progress">Progress percentage in range [0..1].</param>
    /// <param name="loss">Current average loss.</param>
    /// <param name="wst">Words per thread per second.</param>
    /// <param name="eta">Estimated time in seconds.</param>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void TrainProgressCallback(float progress, float loss, double wst, double lr, long eta);
    
    /// <summary>
    /// Autotune progress callback.
    /// </summary>
    /// <param name="progress">Progress percentage in range [0..1].</param>
    /// <param name="trials">Number of trials.</param>
    /// <param name="bestScore">Best score so far.</param>
    /// <param name="eta">Estimated time in seconds.</param>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void AutotuneProgressCallback(double progress, int trials, double bestScore, double eta);
}