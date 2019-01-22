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
            using (var fastText = new FastTextWrapper())
            {

                //fastText.Train(@"C:\Models\cooking.stackexchange.txt", @"C:\Models\cooking", TrainingArgs.SupervisedDefaults(x =>
                //{
                //    x.Epochs = 25;
                //    x.LearningRate = 1.0;
                //    x.WordNGrams = 3;
                //    x.Verbose = 2;
                //    x.LabelPrefix = "__label__";
                //}));

                fastText.LoadModel(@"C:\Models\cooking.bin");
                var labels = fastText.GetLabels();
                var prediction = fastText.PredictSingle("Can I use a larger crockpot than the recipe calls for?");
                var predictions = fastText.PredictMultiple("Can I use a larger crockpot than the recipe calls for?", 4);
                var vector = fastText.GetSentenceVector("Can I use a larger crockpot than the recipe calls for?");
            }
        }
    }
}
