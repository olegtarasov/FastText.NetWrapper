using FastText.NetWrapper;
using Microsoft.Extensions.Logging;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Serilog;
using Serilog.Extensions.Logging;
using Serilog.Sinks.SystemConsole.Themes;
using ShellProgressBar;

namespace ConsoleTest;

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

        using (var pBar = new ProgressBar(100, "Training"))
        {
            var ftArgs = new SupervisedArgs
            {
                epoch = 15,
                lr = 1,
                dim = 300,
                wordNgrams = 2,
                minn = 3,
                maxn = 6,
                verbose = 0,
                TrainProgressCallback = (progress, loss, wst, lr, eta) =>
                {
                    pBar.Tick((int)Math.Ceiling(progress * 100), $"Loss: {loss:N3}, words/thread/sec: {wst}, LR: {lr:N5}, ETA: {eta}");
                }
            };

            fastText.Supervised("cooking.train.txt", outPath, ftArgs);
        }

        try
        {
            File.Delete("_debug.txt");
        }
        catch
        {
        }
            
        log.Information("Validating model on the test set");
            
        var result = fastText.TestInternal("cooking.valid.txt", 1, 0.0f, true);
            
        log.Information($"Results:\n\tPrecision: {result.GlobalMetrics.GetPrecision()}" +
                        $"\n\tRecall: {result.GlobalMetrics.GetRecall()}" +
                        $"\n\tF1: {result.GlobalMetrics.GetF1()}");
            
        var curve = result.GetPrecisionRecallCurve();
        var (_, debugCurve) = TestResult.LoadDebugResult("_debug.txt", fastText.GetLabels());
            
        string plotPath = PlotCurves(tempDir, new []{curve, debugCurve});

        log.Information($"Precision-Recall plot: {plotPath}");

        Console.WriteLine("\nPress any key to exit.");
        Console.ReadKey();
            
        Directory.Delete(tempDir, true);
    }
        
    private static string PlotCurves(string dir, (double precision, double recall)[][] series, string fileName = "precision-recall.svg")
    {
        var plotModel = new PlotModel
        {
            Axes =
            {
                new LinearAxis {Position = AxisPosition.Bottom, Title = "Recall"},
                new LinearAxis {Position = AxisPosition.Left, Title = "Precision"}
            }
        };

        foreach (var serie in series)
        {
            var s = new LineSeries {StrokeThickness = 1};
            s.Points.AddRange(serie.Select(x => new DataPoint(x.recall, x.precision)).OrderBy(x => x.X));
            plotModel.Series.Add(s);
        }
            
        var plotPath = Path.Combine(dir, fileName);
        using (var stream = new FileStream(plotPath, FileMode.Create, FileAccess.Write))
        {
            SvgExporter.Export(plotModel, stream, 600, 600, false);   
        }
            
        return plotPath;
    }
}