using System;
using System.IO;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using Serilog.Sinks.SystemConsole.Themes;
using FastText.NetWrapper;

namespace NugetConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Console(theme: ConsoleTheme.None)
                .CreateLogger();
            
            var log = Log.ForContext<Program>();
            var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);
            
            log.Information($"Temp dir: {tempDir}");
            
            string outPath = Path.Combine(tempDir, "cooking.bin");
            var fastText = new FastTextWrapper(loggerFactory: new LoggerFactory(new[] {new SerilogLoggerProvider()}));

            var ftArgs = new SupervisedArgs
            {
                epoch = 15,
                lr = 1,
                dim = 300,
                wordNgrams = 2,
                minn = 3,
                maxn = 6
            };
            fastText.Supervised("cooking.train.txt",  outPath, ftArgs);
        }
    }
}