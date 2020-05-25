using System;
using System.IO;
using System.Linq;
using FastText.NetWrapper;
using FluentAssertions;
using MartinCostello.Logging.XUnit;
using Microsoft.Extensions.Logging;
using Xunit;
using Xunit.Abstractions;

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
        public void CanGetDefaultArgs()
        {
            var args = new FastTextArgs();

            args.lr.Should().BeApproximately(0.05d, 10e-5);
            args.bucket.Should().Be(2000000);
            args.dim.Should().Be(100);
            args.loss.Should().Be(LossName.NegativeSampling);
            args.model.Should().Be(ModelName.SkipGram);
            args.LabelPrefix.Should().Be("__label__");

            // No need to check all of them
        }

        [Fact]
        public void CanGetDefaultSupervisedArgs()
        {
            var args = FastTextArgs.SupervisedDefaults();
            
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
        public void CanTrainModelWithSuperOldApi()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Train("cooking.train.txt",  outPath, new SupervisedArgs());

            CheckLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainModelWithOldApi()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Train("cooking.train.txt",  outPath, new FastTextArgs());

            CheckLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }
        
        [Fact]
        public void CanTrainSupervised()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            string outPath = Path.Combine(_tempDir, "cooking");
            fastText.Supervised("cooking.train.txt",  outPath, FastTextArgs.SupervisedDefaults());

            CheckLabels(fastText.GetLabels());

            File.Exists(outPath + ".bin").Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        [Fact]
        public void CanLoadSupervisedModel()
        {
            var fastText = new FastTextWrapper(loggerFactory: _loggerFactory);
            fastText.LoadModel(_fixture.ModelPath);

            CheckLabels(fastText.GetLabels());
        }

        [Fact]
        public void CanGetNearestNeighbours()
        {
            var nn = _fixture.FastText.GetNN("train", 5);
            
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
            fastText.Supervised("data.rus.txt",  outPath, FastTextArgs.SupervisedDefaults());

            var labels = fastText.GetLabels();
            labels.Length.Should().Be(2);
            labels.Should().Contain(new[] {"__label__оператор", "__label__выход"});

            var pred = fastText.PredictSingle("Позови оператора");
            pred.Probability.Should().BeGreaterThan(0);
            pred.Label.Should().Be("__label__оператор");

            var nn = fastText.GetNN("оператор", 2);
            nn.Length.Should().Be(2);
            nn.Select(x => x.Label).Should().Contain(new[] {"на", "поговорить"});
            foreach (var prediction in nn)
            {
                prediction.Probability.Should().BeGreaterThan(0);
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