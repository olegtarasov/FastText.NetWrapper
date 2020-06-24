using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
            var args = new QuantizeArgs();

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

            // No need to check all of them
        }
        
        // Deprecated methods removed.
        // [Fact]
        // public void CanTrainModelWithSuperOldApi()
        // {
        //     var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
        //     string outPath = Path.Combine(_tempDir, "cooking");
        //     fastText.Train("cooking.train.txt",  outPath, new SupervisedArgs());
        //
        //     CheckLabels(fastText.GetLabels());
        //
        //     File.Exists(outPath + ".bin").Should().BeTrue();
        //     File.Exists(outPath + ".vec").Should().BeTrue();
        // }

        // [Fact]
        // public void CanTrainModelWithOldApi()
        // {
        //     var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
        //     string outPath = Path.Combine(_tempDir, "cooking");
        //     fastText.Train("cooking.train.txt", outPath, new SupervisedArgs());
        //
        //     CheckLabels(fastText.GetLabels());
        //
        //     File.Exists(outPath + ".bin").Should().BeTrue();
        //     File.Exists(outPath + ".vec").Should().BeTrue();
        // }

        [Fact]
        public void CanTrainSupervised()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Supervised("cooking.train.txt",  outPath, new SupervisedArgs());

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            CheckLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainSupervisedWithNoLoggingAndNoArgs()
        {
            var fastText = new FastTextWrapper();
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Supervised("cooking.train.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            CheckLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        [Fact]
        public void CanLoadSupervisedModel()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            fastText.LoadModel(_fixture.ModelPath);
            
            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            CheckLabels(fastText.GetLabels());
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
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
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
            var nn = fastText.GetNN("оператор", 2);
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
            var fastText = new FastTextWrapper();

            fastText.IsModelReady().Should().BeFalse();
        }

        [Fact]
        public void CantUsePretrainedVectorsWithDifferentDimension()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            
            string outPath = Path.Combine(_tempDir, "cooking");
            var args = new SupervisedArgs();
            args.PretrainedVectors = "cooking.unsup.300.vec";

            fastText.Invoking(x => x.Supervised("cooking.train.txt", outPath, args))
                .Should().Throw<NativeLibraryException>()
                .WithMessage("Dimension of pretrained vectors (300) does not match dimension (100)!");
        }
        
        [Fact]
        public void CanUsePretrainedVectorsForSupervisedModel()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            
            string outPath = Path.Combine(_tempDir, "cooking");
            var args = new SupervisedArgs();
            args.PretrainedVectors = "cooking.unsup.300.vec";
            args.dim = 300;

            fastText.Supervised("cooking.train.txt", outPath, args);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(300);
            
            CheckLabels(fastText.GetLabels());
            
            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
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

        [Fact]
        public void CanTrainSkipgramModel()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Unsupervised(UnsupervisedModel.SkipGram, "cooking.train.nolabels.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainCbowModel()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Unsupervised(UnsupervisedModel.CBow, "cooking.train.nolabels.txt",  outPath);

            fastText.IsModelReady().Should().BeTrue();
            fastText.GetModelDimension().Should().Be(100);

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        [Fact]
        public void SkipgramAndCBowLearnDifferentRepresentations()
        {
            var sg = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outSG = Path.Combine(_tempDir, "cooking");
            sg.Unsupervised(UnsupervisedModel.SkipGram, "cooking.train.nolabels.txt",  outSG);
            
            var cbow = new FastTextWrapper(loggerFactory: _loggerFactory);
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


        private void CheckLabels(string[] modelLabels)
        {
            modelLabels.Length.Should().Be(_labels.Length);
            modelLabels.Should().Contain(_labels);
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