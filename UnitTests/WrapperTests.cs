using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using FastText.NetWrapper;
using FluentAssertions;
using MartinCostello.Logging.XUnit;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;
using Xunit;
using Xunit.Abstractions;
using TestResult = FastText.NetWrapper.TestResult;

namespace UnitTests
{
    public class WrapperTests : IDisposable, IClassFixture<SupervisedModelFixture>
    {
        private readonly string _tempDir;
        private readonly ILoggerFactory _loggerFactory;
        private readonly ILogger<WrapperTests> _logger;
        private readonly SupervisedModelFixture _fixture;
        private readonly string[] _labels;

        public WrapperTests(ITestOutputHelper helper, SupervisedModelFixture fixture)
        {
            _fixture = fixture;
            _loggerFactory = new LoggerFactory(new []{new XUnitLoggerProvider(helper, new XUnitLoggerOptions())});
            _logger = _loggerFactory.CreateLogger<WrapperTests>();
            _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            _labels = File.ReadAllLines("labels.txt");
            Directory.CreateDirectory(_tempDir);
            
            _logger.LogInformation($"Temp dir is: {_tempDir}");
        }

        [Fact]
        public void StructureSizesAreCorrect()
        {
            Marshal.SizeOf<FastTextWrapper.FastTextArgsStruct>().Should().Be(100);
            Marshal.SizeOf<FastTextWrapper.AutotuneArgsStruct>().Should().Be(36);
            Marshal.SizeOf<FastTextWrapper.TestMetrics>().Should().Be(48);
            Marshal.SizeOf<FastTextWrapper.TestMeter>().Should().Be(40);
        }

        [Fact]
        public void CanGetDefaultSupervisedArgs()
        {
            var args = new SupervisedArgs();
            
            args.bucket.Should().Be(2000000);
            args.dim.Should().Be(100);
            args.loss.Should().Be(LossName.Softmax);
            args.model.Should().Be(ModelName.Supervised);
            args.minCount.Should().Be(1);
            args.minn.Should().Be(0);
            args.maxn.Should().Be(0);
            args.lr.Should().BeApproximately(0.1d, 10e-5);
        }

        [Fact]
        public void CanGetDefaultQuantizeArgs()
        {
            var args = new QuantizedSupervisedArgs();

            args.dsub.Should().Be(2);
        }
        
        [Fact]
        public void CanGetDefaultSkipgramArgs()
        {
            var args = new UnsupervisedArgs();

            args.lr.Should().BeApproximately(0.05d, 10e-5);
            args.bucket.Should().Be(2000000);
            args.dim.Should().Be(100);
            args.loss.Should().Be(LossName.NegativeSampling);
            args.model.Should().Be(ModelName.SkipGram);
            args.LabelPrefix.Should().Be("__label__");

            // No need to check all of them here.
        }
        

        [Fact]
        public void CanTrainSupervised()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            
            var args = new SupervisedArgs();
            var tuneArgs = new AutotuneArgs();
            
            fastText.Supervised("cooking.train.txt",  outPath, args, tuneArgs, true);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".bin");

            AssertLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();

            var debugArgs = DebugArgs.Load("_train.txt");
            AssertSupervisedArgs(args, debugArgs.ExternalArgs);
            AssertSupervisedArgs(args, debugArgs.ConvertedArgs);
            AssertAutotuneArgs(tuneArgs, debugArgs.ExternalTune);
            AssertAutotuneArgs(tuneArgs, debugArgs.ConvertedTune);

            debugArgs.ExternalInput.Should().Be("cooking.train.txt");
            debugArgs.ConvertedInput.Should().Be("cooking.train.txt");
            debugArgs.ExternalOutput.Should().Be(outPath);
            debugArgs.ConvertedOutput.Should().Be(outPath);
        }
        
        [Fact]
        public void CanTrainSupervisedAndQuantize()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            
            var args = new QuantizedSupervisedArgs();
            var tuneArgs = new AutotuneArgs();
            
            fastText.Supervised("cooking.train.txt", outPath, args, tuneArgs, true);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".ftz");

            AssertLabels(fastText.GetLabels());

            File.Exists(outPath + ".ftz").Should().BeTrue();
            File.Exists(outPath + ".bin").Should().BeFalse();
            File.Exists(outPath + ".vec").Should().BeFalse();

            var debugArgs = DebugArgs.Load("_train.txt");
            AssertSupervisedArgs(args, debugArgs.ExternalArgs);
            AssertSupervisedArgs(args, debugArgs.ConvertedArgs);
            AssertAutotuneArgs(tuneArgs, debugArgs.ExternalTune);
            AssertAutotuneArgs(tuneArgs, debugArgs.ConvertedTune);

            debugArgs.ExternalInput.Should().Be("cooking.train.txt");
            debugArgs.ConvertedInput.Should().Be("cooking.train.txt");
        }
        
        [Fact]
        public void CanTrainSupervisedWithRelativeOutput()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            
            var args = new SupervisedArgs();
            var tuneArgs = new AutotuneArgs();
            
            fastText.Supervised("cooking.train.txt",  "cooking", args, tuneArgs, true);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be("cooking.bin");

            AssertLabels(fastText.GetLabels());

            File.Exists("cooking.bin").Should().BeTrue();
            File.Exists("cooking.vec").Should().BeTrue();

            File.Delete("cooking.bin");
            File.Delete("cooking.vec");
        }

        [Fact]
        public void CanTrainSupervisedWithNoLoggingAndNoArgs()
        {
            using var fastText = new FastTextWrapper();
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Supervised("cooking.train.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".bin");

            AssertLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainSupervisedWithProgressCallback()
        {
            using var fastText = new FastTextWrapper();
            string outPath = Path.Combine(_tempDir, "cooking");
            int callNum = 0;

            var args = new SupervisedArgs
            {
                TrainProgressCallback = (progress, loss, wst, lr, eta) =>
                {
                    callNum++;
                }
            };
            
            fastText.Supervised("cooking.train.txt",  outPath, args);

            callNum.Should().BeGreaterThan(0);
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".bin");

            AssertLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        [Fact]
        public void CantTrainSupervisedWithPretrainedVectorsWithDifferentDimension()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            
            string outPath = Path.Combine(_tempDir, "cooking");
            var args = new SupervisedArgs {PretrainedVectors = "cooking.unsup.300.vec"};

            fastText.Invoking(x => x.Supervised("cooking.train.txt", outPath, args))
                .Should().Throw<NativeLibraryException>()
                .WithMessage("Dimension of pretrained vectors (300) does not match dimension (100)!");
        }

        [Fact]
        public void CanTrainSupervisedWithPretrainedVectors()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            
            string outPath = Path.Combine(_tempDir, "cooking");
            var args = new SupervisedArgs();
            args.PretrainedVectors = "cooking.unsup.300.vec";
            args.dim = 300;
        
            fastText.Supervised("cooking.train.txt", outPath, args, new AutotuneArgs(), true);
        
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(300);
            fastText.ModelPath.Should().Be(outPath + ".bin");
            
            AssertLabels(fastText.GetLabels());
            
            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainSkipgramModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Unsupervised(UnsupervisedModel.SkipGram, "cooking.train.nolabels.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".bin");

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainCbowModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Unsupervised(UnsupervisedModel.CBow, "cooking.train.nolabels.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(outPath + ".bin");

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        [Fact]
        public void SkipgramAndCBowLearnDifferentRepresentations()
        {
            using var sg = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outSG = Path.Combine(_tempDir, "cooking");
            sg.Unsupervised(UnsupervisedModel.SkipGram, "cooking.train.nolabels.txt",  outSG);
            
            using var cbow = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outCbow = Path.Combine(_tempDir, "cooking");
            cbow.Unsupervised(UnsupervisedModel.CBow, "cooking.train.nolabels.txt",  outCbow);

            var nnSg = sg.GetNearestNeighbours("pot", 10);
            var nnCbow = cbow.GetNearestNeighbours("pot", 10);
            var nnSup = _fixture.FastText.GetNearestNeighbours("pot", 10);

            void CheckPair(Prediction[] first, Prediction[] second)
            {
                int samePredictions = 0;

                foreach (var prediction in first)
                {
                    if (second.Any(x => x.Label == prediction.Label))
                        samePredictions++;
                }

                // We want less than a half of same predictions.
                samePredictions.Should().BeLessThan(first.Length / 2);
            }
            
            CheckPair(nnSg, nnCbow);
            CheckPair(nnSg, nnSup);
            CheckPair(nnCbow, nnSup);
        }

        [Fact]
        public void CanAutotuneSupervisedModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");

            var args = new SupervisedArgs
            {
                bucket = 2100000,
                dim = 250,
                epoch = 10,
                loss = LossName.HierarchicalSoftmax,
                lr = 0.5,
                maxn = 5,
                minn = 2,
                neg = 6,
                seed = 42,
                t = 0.0002,
                thread = 10,
                verbose = 1,
                ws = 6,
                minCount = 2,
                saveOutput = true,
                wordNgrams = 2,
                lrUpdateRate = 110,
                minCountLabel = 1
            };
            
            var autotuneArgs = new AutotuneArgs
            {
                Duration = 30,
                Metric = "precisionAtRecall:30",
                Predictions = 2,
                ValidationFile = "cooking.valid.txt"
            };
            
            fastText.Supervised("cooking.train.txt",  outPath, args, autotuneArgs, true);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(250);
            fastText.ModelPath.Should().Be(outPath + ".bin");
            
            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();

            var debugArgs = DebugArgs.Load("_train.txt");
            
            AssertSupervisedArgs(args, debugArgs.ExternalArgs);
            AssertSupervisedArgs(args, debugArgs.ConvertedArgs);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ExternalTune);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ConvertedTune);

            debugArgs.ExternalInput.Should().Be("cooking.train.txt");
            debugArgs.ConvertedInput.Should().Be("cooking.train.txt");
            debugArgs.ExternalOutput.Should().Be(outPath);
            debugArgs.ConvertedOutput.Should().Be(outPath);
        }
        
        [Fact]
        public void CanAutotuneSupervisedModelWithProgressCallback()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            int numCalls = 0;

            var args = new SupervisedArgs
            {
                bucket = 2100000,
                dim = 250,
                epoch = 10,
                loss = LossName.HierarchicalSoftmax,
                lr = 0.5,
                maxn = 5,
                minn = 2,
                neg = 6,
                seed = 42,
                t = 0.0002,
                thread = 10,
                verbose = 1,
                ws = 6,
                minCount = 2,
                saveOutput = true,
                wordNgrams = 2,
                lrUpdateRate = 110,
                minCountLabel = 1
            };
            
            var autotuneArgs = new AutotuneArgs
            {
                Duration = 30,
                Metric = "precisionAtRecall:30",
                Predictions = 2,
                ValidationFile = "cooking.valid.txt",
                AutotuneProgressCallback = (progress, trials, score, eta) =>
                {
                    numCalls++;
                }
            };
            
            fastText.Supervised("cooking.train.txt",  outPath, args, autotuneArgs, true);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(250);
            fastText.ModelPath.Should().Be(outPath + ".bin");
            
            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();

            var debugArgs = DebugArgs.Load("_train.txt");
            
            AssertSupervisedArgs(args, debugArgs.ExternalArgs);
            AssertSupervisedArgs(args, debugArgs.ConvertedArgs);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ExternalTune);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ConvertedTune);

            debugArgs.ExternalInput.Should().Be("cooking.train.txt");
            debugArgs.ConvertedInput.Should().Be("cooking.train.txt");
            debugArgs.ExternalOutput.Should().Be(outPath);
            debugArgs.ConvertedOutput.Should().Be(outPath);
        }

        [Fact]
        public void CanAutotuneQuantizedSupervisedModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");

            var args = new QuantizedSupervisedArgs
            {
                bucket = 2100000,
                dim = 250,
                epoch = 10,
                loss = LossName.HierarchicalSoftmax,
                lr = 0.5,
                maxn = 5,
                minn = 2,
                neg = 6,
                seed = 42,
                t = 0.0002,
                thread = 10,
                verbose = 1,
                ws = 6,
                minCount = 2,
                saveOutput = true,
                wordNgrams = 2,
                lrUpdateRate = 110,
                minCountLabel = 1,

                cutoff = 10000,
                dsub = 3,
                retrain = true
            };
            
            var autotuneArgs = new AutotuneArgs
            {
                Duration = 60,
                Metric = "precisionAtRecall:30",
                Predictions = 2,
                ModelSize = "10M",
                ValidationFile = "cooking.valid.txt"
            };
            
            fastText.Supervised("cooking.train.txt",  outPath, args, autotuneArgs, true);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(250);
            fastText.ModelPath.Should().Be(outPath + ".ftz");
            
            File.Exists(outPath + ".ftz").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();


            var debugArgs = DebugArgs.Load("_train.txt");
            
            AssertQuantizedArgs(args, debugArgs.ExternalArgs);
            AssertQuantizedArgs(args, debugArgs.ConvertedArgs);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ExternalTune);
            AssertAutotuneArgs(autotuneArgs, debugArgs.ConvertedTune);

            debugArgs.ExternalInput.Should().Be("cooking.train.txt");
            debugArgs.ConvertedInput.Should().Be("cooking.train.txt");
            debugArgs.ExternalOutput.Should().Be(outPath);
            debugArgs.ConvertedOutput.Should().Be(outPath);
        }

        [Fact]
        public void CanLoadSupervisedModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            fastText.LoadModel(_fixture.FastText.ModelPath);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            AssertLabels(fastText.GetLabels());
        }

        [Fact]
        public void CanQuantizeLoadedSupervisedModel()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            fastText.LoadModel(_fixture.FastText.ModelPath);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            AssertLabels(fastText.GetLabels());
            
            string newPath = Path.Combine(Path.GetDirectoryName(_fixture.FastText.ModelPath), Path.GetFileNameWithoutExtension(_fixture.FastText.ModelPath));
            
            fastText.Quantize();

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);
            fastText.ModelPath.Should().Be(newPath + ".ftz");
            
            File.Exists(newPath + ".ftz").Should().BeTrue();
            File.Exists(newPath + ".vec").Should().BeTrue();

        }

        [Fact]
        public void CanGetNearestNeighbours()
        {
            var nn = _fixture.FastText.GetNearestNeighbours("train", 5);
            
            nn.Length.Should().Be(5);

            foreach (var prediction in nn)
            {
                prediction.Probability.Should().BeGreaterThan(0);
                prediction.Label.Should().NotBeNullOrEmpty();
            }
        }

        [Fact]
        public void CanGetSentenceVector()
        {
            var vec = _fixture.FastText.GetSentenceVector("This is just a test");

            vec.Length.Should().Be(100);
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i].Should().NotBe(0);
            }
        }

        [Fact]
        public void CanGetWordVector()
        {
            var vec = _fixture.FastText.GetWordVector("pot");

            vec.Length.Should().Be(100);
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i].Should().NotBe(0);
            }
        }

        [Fact]
        public void CanPredictSingle()
        {
            var pred = _fixture.FastText.PredictSingle("How to boil a goose?");
            pred.Probability.Should().BeGreaterThan(0);
            _labels.Should().Contain(pred.Label);
        }

        [Fact]
        public void CanPredictMultiple()
        {
            var preds = _fixture.FastText.PredictMultiple("How to boil a goose?", 5);

            preds.Length.Should().Be(5);
            foreach (var pred in preds)
            {
                pred.Probability.Should().BeGreaterThan(0);
                _labels.Should().Contain(pred.Label);
            }
        }

        [Fact]
        public void CanHandleUtf8()
        {
            using var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "rus");
            fastText.Supervised("data.rus.txt",  outPath, new SupervisedArgs());

            var labels = fastText.GetLabels();
            labels.Length.Should().Be(2);
            labels.Should().Contain(new[] {"__label__оператор", "__label__выход"});

            var pred = fastText.PredictSingle("Позови оператора");
            pred.Probability.Should().BeGreaterThan(0);
            pred.Label.Should().Be("__label__оператор");

            var sourceWords = File.ReadAllText("data.rus.txt")
                .Split(new[] {" ", "\r\n", "\n"}, StringSplitOptions.RemoveEmptyEntries)
                .Where(x => !x.StartsWith("__label__"))
                .Distinct().ToArray();
            var nn = fastText.GetNearestNeighbours("оператор", 2);
            nn.Length.Should().Be(2);
            sourceWords.Should().Contain(nn.Select(x => x.Label));
            foreach (var prediction in nn)
            {
                prediction.Probability.Should().BeGreaterThan(0);
            }
        }

        [Fact]
        public void EmptyModelIsNotReady()
        {
            using var fastText = new FastTextWrapper();

            fastText.IsModelReady().Should().BeFalse();
        }

        [Fact]
        public void CanTestSupervisedModel()
        {
            var result = _fixture.FastText.TestInternal("cooking.valid.txt", 1, 0.0f, true);
            var (debugResult, debugCurve) = TestResult.LoadDebugResult("_debug.txt", _fixture.FastText.GetLabels());

            result.Examples.Should().Be(debugResult.Examples);
            AssertMetrics(result.GlobalMetrics, debugResult.GlobalMetrics);
            result.LabelMetrics.Count.Should().Be(debugResult.LabelMetrics.Count);

            foreach (var metrics in result.LabelMetrics.Values)
                AssertMetrics(metrics, debugResult.LabelMetrics[metrics.Label]);

            var curve = result.GetPrecisionRecallCurve();
            curve.Length.Should().Be(debugCurve.Length);

            for (int i = 0; i < curve.Length; i++)
            {
                curve[i].precision.Should().BeApproximately(debugCurve[i].precision, 10e-5);
                curve[i].recall.Should().BeApproximately(debugCurve[i].recall, 10e-5);
            }
        }



        private void AssertMetrics(Metrics actual, Metrics expected)
        {
            actual.Gold.Should().Be(expected.Gold);
            actual.Predicted.Should().Be(expected.Predicted);
            actual.PredictedGold.Should().Be(expected.PredictedGold);
            actual.ScoreVsTrue.Length.Should().Be(expected.ScoreVsTrue.Length);

            for (int i = 0; i < actual.ScoreVsTrue.Length; i++)
            {
                actual.ScoreVsTrue[i].score.Should().BeApproximately(expected.ScoreVsTrue[i].score, 10e-5f);
                actual.ScoreVsTrue[i].gold.Should().BeApproximately(expected.ScoreVsTrue[i].gold, 10e-5f);
            }
        }
        
        private void AssertLabels(string[] modelLabels)
        {
            modelLabels.Length.Should().Be(_labels.Length);
            modelLabels.Should().Contain(_labels);
        }
        
        private void AssertSupervisedArgs(SupervisedArgs expected, SupervisedArgs actual)
        {
            actual.lr.Should().Be(expected.lr);
            actual.lrUpdateRate.Should().Be(expected.lrUpdateRate);
            actual.dim.Should().Be(expected.dim);
            actual.ws.Should().Be(expected.ws);
            actual.epoch.Should().Be(expected.epoch);
            actual.minCount.Should().Be(expected.minCount);
            actual.minCountLabel.Should().Be(expected.minCountLabel);
            actual.neg.Should().Be(expected.neg);
            actual.wordNgrams.Should().Be(expected.wordNgrams);
            actual.loss.Should().Be(expected.loss);
            actual.model.Should().Be(expected.model);
            actual.bucket.Should().Be(expected.bucket);
            actual.minn.Should().Be(expected.minn);
            actual.maxn.Should().Be(expected.maxn);
            actual.thread.Should().Be(expected.thread);
            actual.t.Should().Be(expected.t);
            (actual.LabelPrefix ?? "").Should().Be(expected.LabelPrefix ?? "");
            actual.verbose.Should().Be(expected.verbose);
            (actual.PretrainedVectors ?? "").Should().Be(expected.PretrainedVectors ?? "");
            actual.saveOutput.Should().Be(expected.saveOutput);
            actual.seed.Should().Be(expected.seed);
        }

        private void AssertQuantizedArgs(QuantizedSupervisedArgs expected, QuantizedSupervisedArgs actual)
        {
            AssertSupervisedArgs(expected, actual);

            actual.qout.Should().Be(expected.qout);
            actual.retrain.Should().Be(expected.retrain);
            actual.qnorm.Should().Be(expected.qnorm);
            actual.cutoff.Should().Be(expected.cutoff);
            actual.dsub.Should().Be(expected.dsub);
        }

        private void AssertAutotuneArgs(AutotuneArgs expected, AutotuneArgs actual)
        {
            (actual.ValidationFile ?? "").Should().Be(expected.ValidationFile ?? "");
            (actual.Metric ?? "").Should().Be(expected.Metric ?? "");
            actual.Predictions.Should().Be(expected.Predictions);
            actual.Duration.Should().Be(expected.Duration);
            (actual.ModelSize ?? "").Should().Be(expected.ModelSize ?? "");
        }

        public void Dispose()
        {
            try
            {
                Directory.Delete(_tempDir, true);
            }
            catch
            {
            }
        }
    }
}