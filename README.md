[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/olegtarasov/FastText.NetWrapper/Build%20and%20publish?style=flat-square)](https://github.com/olegtarasov/FastText.NetWrapper/actions)
[![Nuget](https://img.shields.io/nuget/v/FastText.NetWrapper?style=flat-square)](https://www.nuget.org/packages/FastText.NetWrapper)
[![Donwloads](https://img.shields.io/nuget/dt/FastText.NetWrapper?label=Nuget&style=flat-square)](https://www.nuget.org/packages/FastText.NetWrapper)

# FastText.NetWrapper

This is a cross-platform .NET Standard wrapper for Facebook's [FastText](https://github.com/facebookresearch/fastText) library. 
The wrapper comes with bundled precompiled native binaries for all three platforms: Windows, Linux and MacOs.

Just add it to your project and start using it! No additional setup required. This library will unpack and call appropriate native 
binary depending on target platform.

## What's new

### Version 1.2.0

Version 1.2.0 introduces a few breaking changes to library API. If you are not ready to migrate, use v. `1.1.0`.

* **❗️Breaking change:️** Removed both deprecated `Train()` methods.
* **❗️Breaking change:️** Removed deprecated `SupervisedArgs` class.
* **❗️Breaking change:️** Removed `FastTextArgs.SupervisedDefaults()` in favor of new `SupervisedArgs` with default constructor.
* **❗️Breaking change:️** `FastTextArgs` class can't be constructed directly, use new `SupervisedArgs` and `UnsupervisedArgs` classes.
* Added an `Unsupervised()` method to train Skipgram or Cbow models.

### Version 1.1.0

* Added new `Supervised()` method as part of streamlining the API.
* Added new `Test()` method for testing supervised model.
* Deprecated both `Train()` methods. They will be removed in v. `1.2.0`.

## Version `1.2.0` migration guide

* Instead of old `Train()` methods use `Supervised()` and `Unsupervised()` methods.
* Instead of `FastTextArgs.SupervisedDefaults()` use `SupervisedArgs` or `Supervised()` overload with 2 arguments.

## Usage

Library API closely follows fastText command-line interface, so you can jump right in.

### Supervised model training

The simplest use case is to train a supervised model with default parameters. We create a `FastTextWrapper` and call `Supervised()`.

```c#
var fastText = new FastTextWrapper();
fastText.Supervised("cooking.train.txt",  "cooking");
```

Note the arguments:

1. We specify an input file with one labeled example per line. Here we use Stack Overflow cooking dataset from Facebook:
https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz. You can find extracted files split into training
and validation sets in `UnitTests` directory in this repository.
2. Your model will be saved to `cooking.bin` and `cooking.vec` with pretrained vectors will be placed if the same directory.
3. Here we use `Supervised()` overload with 2 arguments. This means that training will be done with default parameters. 
It's a good starting point and is the same as calling fastText this way:

```bash
./fasttext supervised -input cooking.train.txt -output cooking
```

### Loading models

Call `LoadModel()` and specify path to the `.bin` model file:

```c#
var fastText = new FastTextWrapper();
fastText.LoadModel("model.bin");
```

### Using pretrained vectors

To use pretrained vectors for your supervised model, create an instance of `SupervisedArgs` and customize it:

```c#
var fastText = new FastTextWrapper();
            
var args = new SupervisedArgs
{
    PretrainedVectors = "cooking.unsup.300.vec",
    dim = 300
};

fastText.Supervised("cooking.train.txt", "cooking", args);
```

Here we get default training arguments, supply a path to pretrained vectors file and adjust vector dimension accordingly.

**Important!** Be sure to always check the dimension of your pretrained vectors! Many vectors on the internet have dimension `300`,
but default dimension for fastText supervised model training is `100`.

### Testing the model

Now you can easily test a supervised model against a validation set. You can specify different values for `k` and `threshlod` as well.

```c#
var result = fastText.Test("cooking.valid.txt");
```

You will get an instance of `TestResult` where you can find aggregated or per-label metrics:

```c#
Console.WriteLine($"Results:\n\tPrecision: {result.GlobalMetrics.GetPrecision()}" +
                            $"\n\tRecall: {result.GlobalMetrics.GetRecall()}" +
                            $"\n\tF1: {result.GlobalMetrics.GetF1()}");
```

You can even get a precision-recall curve (aggregated or per-label)! Here is an example of exporting an SVG plot with cross-platform
[OxyPlot library](https://oxyplot.github.io):

```c#
var result = fastText.Test("cooking.valid.txt");
var curve = result.GetPrecisionRecallCurve();

var series = new LineSeries {StrokeThickness = 1};
series.Points.AddRange(curve.Select(x => new DataPoint(x.recall, x.precision)).OrderBy(x => x.X));

var plotModel = new PlotModel
{
    Series = { series },
    Axes =
    {
        new LinearAxis {Position = AxisPosition.Bottom, Title = "Recall"},
        new LinearAxis {Position = AxisPosition.Left, Title = "Precision"}
    }
};

using (var stream = new FileStream("precision-recall.svg", FileMode.Create, FileAccess.Write))
{
    SvgExporter.Export(plotModel, stream, 600, 600, false);   
}
```

![](docs/prec-rec.png)

### Training unsupervised models

Use `Unsupervised()` method specifying model type: Skipgram or Cbow:

```c#
var fastText = new FastTextWrapper();
fastText.Unsupervised(UnsupervisedModel.SkipGram, "cooking.train.nolabels.txt",  "cooking");
```

You can use an optional `UnsupervisedArgs` argument to customize training.

### Automatic hyperparameter tuning

Coming soon!

### Getting logs from the wrapper

`FastTextWrapper` can produce a small amount of logs mostly concerning native library management. You can turn logging on by providing an
instance of `Microsoft.Extensions.Logging.ILoggerFactory`. In this example we use Serilog with console sink.

You can also inject your standard `IloggerFactory` through .NET Core DI.

```c#
Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Console(theme: ConsoleTheme.None)
                .CreateLogger();

var fastText = new FastTextWrapper(loggerFactory: new LoggerFactory(new[] {new SerilogLoggerProvider()}));
```

### Handling native exceptions

In version `1.1` I've added much better native error handling. Now in case of most native errors you will get a nice
`NativeLibraryException` which you can inspect for detailed error description.

## Windows Requirements

Since this wrapper uses native C++ binaries under the hood, you will need to have Visual C++ Runtime Version 140 installed when 
running under Windows. Visit the MS Downloads page (https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) 
and select the appropriate redistributable. 

## FastText C-style API

If you are interested in using FastText with C-style API, here is my fork of the official library: https://github.com/olegtarasov/fastText.