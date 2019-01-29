using System;
using System.Collections.Generic;
using System.IO;
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
                TrainSupervised();
                //TrainLowLevel();
                LoadModel();
            }
        }

        private static void LoadModel()
        {
            using (var fastText = new FastTextWrapper())
            {
                fastText.LoadModel(@"D:\__Models\cooking.bin");
                var labels = fastText.GetLabels();
                var prediction = fastText.PredictSingle("Can I use a larger crockpot than the recipe calls for?");
                var predictions = fastText.PredictMultiple("Can I use a larger crockpot than the recipe calls for?", 4);
                var vector = fastText.GetSentenceVector("Can I use a larger crockpot than the recipe calls for?");
            }
        }

        private static void TrainLowLevel()
        {
            using (var fastText = new FastTextWrapper())
            {
                fastText.Train(@"D:\__Models\cooking.train.txt", @"D:\__Models\cooking", new FastTextArgs
                                                                                     {

                                                                                     });
            }
        }

        private static void TrainSupervised()
        {
            using (var fastText = new FastTextWrapper())
            {
                fastText.Train(@"D:\__Models\cooking.train.txt", @"D:\__Models\cooking", SupervisedArgs.SupervisedDefaults(x =>
                {
                    x.Epochs = 25;
                    x.LearningRate = 1.0;
                    x.WordNGrams = 3;
                    x.Verbose = 2;
                    x.LabelPrefix = "__label__";
                }));
            }
        }
    }
}