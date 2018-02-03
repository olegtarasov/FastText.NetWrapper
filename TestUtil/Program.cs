using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FastText.NetWrapper;

namespace TestUtil
{
    class Program
    {
        static void Main(string[] args)
        {
            var fastText = new FastTextWrapper();

            fastText.Train(@"C:\_Models\botTest.txt", @"C:\_Models\botModel", TrainingArgs.SupervisedDefaults(x =>
            {
                x.Epochs = 25;
                x.LearningRate = 1.0;
                x.WordNGrams = 3;
                x.Verbose = 2;
            }));

            //fastText.LoadModel(@"C:\_Models\fasttext.bin");
            var prediction = fastText.PredictSingle("отключить услуга гудок");
        }
    }
}
