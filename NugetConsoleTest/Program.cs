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
            
            var ftArgs = FastTextArgs.SupervisedDefaults();
            ftArgs.epoch = 15;
            ftArgs.lr = 1;
            ftArgs.dim = 300;
            ftArgs.wordNgrams = 2;
            ftArgs.minn = 3;
            ftArgs.maxn = 6;
            fastText.Supervised("cooking.train.txt",  outPath, ftArgs);
        }
    }
}