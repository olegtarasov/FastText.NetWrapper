using System;
using System.IO;
using FastText.NetWrapper;
using FluentAssertions;

namespace UnitTests
{
    public class SupervisedModelFixture : IDisposable
    {
        private readonly string _tempDir;

        public SupervisedModelFixture()
        {
            _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempDir);
            
            FastText = new FastTextWrapper();
            string outPath = Path.Combine(_tempDir, "cooking");
            FastText.Supervised("cooking.train.txt",  outPath, FastTextArgs.SupervisedDefaults());

            ModelPath = outPath + ".bin";

            File.Exists(ModelPath).Should().BeTrue();
            File.Exists(outPath + ".vec").Should().BeTrue();
        }

        public FastTextWrapper FastText { get; }
        public string ModelPath { get; set; }

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