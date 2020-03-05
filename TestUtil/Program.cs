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
        private static string Usage = $"Usage: tesutil [train|trainlowlevel|load] train_file model_file{Environment.NewLine}Usage: tesutil nn model_file";
        
        static void Main(string[] args)
        {
            if ((args.FirstOrDefault() == "nn" && args.Length < 2) || (args.FirstOrDefault() != "nn" && args.Length < 3))
            {
                Console.WriteLine(Usage);
                return;
            }

            using (var fastText = new FastTextWrapper())
            {
                switch (args[0])
                {
                    case "train":
                        TrainSupervised(fastText, args[1], args[2]);
                        break;
                    case "trainlowlevel":
                        TrainLowLevel(fastText, args[1], args[2]);
                        break;
                    case "load":
                        fastText.LoadModel(args[2]);
                        break;
                }

                if (args[0] != "nn")
                {
                    Test(fastText);
                }
                else
                {
                    fastText.LoadModel(args[1]);
                    TestNN(fastText);
                }
            }
        }

        private static void TrainLowLevel(FastTextWrapper fastText, string trainFile, string modelFile)
        {
            fastText.Train(trainFile, modelFile, new FastTextArgs
            {

            });
        }

        private static void TrainSupervised(FastTextWrapper fastText, string trainFile, string modelFile)
        {
            fastText.Train(trainFile, modelFile, SupervisedArgs.SupervisedDefaults(
                x =>
                {
                    x.Epochs = 25;
                    x.LearningRate = 1.0;
                    x.WordNGrams = 3;
                    x.Verbose = 2;
                    x.LabelPrefix = "__label__";
                }));
        }

        private static void Test(FastTextWrapper fastText)
        {
            var labels = fastText.GetLabels();
            var prediction = fastText.PredictSingle("Can I use a larger crockpot than the recipe calls for?");
            var predictions = fastText.PredictMultiple("Can I use a larger crockpot than the recipe calls for?", 4);
            var vector = fastText.GetSentenceVector("Can I use a larger crockpot than the recipe calls for?");
        }

        private static void TestNN(FastTextWrapper fastText)
        {
            fastText.GetNN("train", 5);
        }
    }
}