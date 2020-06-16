using System;
using System.IO;
using System.Runtime.InteropServices;

namespace FastText.NetWrapper
{
    public class Metrics
    {
        internal Metrics(FastTextWrapper.TestMetrics metrics, string label)
        {
            Gold = metrics.Gold;
            Predicted = metrics.Predicted;
            PredictedGold = metrics.PredictedGold;
            Label = label;

            ScoreVsTrue = new (float score, float gold)[metrics.ScoresLen];

            if (metrics.ScoresLen == 0)
            {
                return;
            }
            
            var gold = new float[metrics.ScoresLen];
            var pred = new float[metrics.ScoresLen];
            
            Marshal.Copy(metrics.GoldScores, gold, 0, metrics.ScoresLen);
            Marshal.Copy(metrics.PredictedScores, pred, 0, metrics.ScoresLen);

            for (int i = 0; i < metrics.ScoresLen; i++)
            {
                ScoreVsTrue[i] = (pred[i], gold[i]);
            }
        }

        public string Label { get; set; }

        public long Gold { get; set; }

        public long Predicted { get; set; }

        public long PredictedGold { get; set; }

        public (float score, float gold)[] ScoreVsTrue { get; set; }
    }
}