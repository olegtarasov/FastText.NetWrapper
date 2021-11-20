using System.Runtime.InteropServices;

namespace FastText.NetWrapper;

public class Metrics
{
    /// <summary>
    /// Ctor.
    /// </summary>
    public Metrics()
    {
    }

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

    /// <summary>
    /// Metrics label. Will be <code>null</code> if this is a <see cref="TestResult.GlobalMetrics"/>.
    /// </summary>
    public string Label { get; set; }

    /// <summary>
    /// Total number of times this label was a true label in a prediction.
    /// In case of <see cref="TestResult.GlobalMetrics"/> it's a total number of
    /// true labels (not distinct) through all test predictions.
    /// </summary>
    public long Gold { get; set; }

    /// <summary>
    /// Total number of times this label was predicted.
    /// In case of <see cref="TestResult.GlobalMetrics"/> it's a total number of
    /// predicted labels (not distinct) through all predictions.
    /// </summary>
    public long Predicted { get; set; }

    /// <summary>
    /// Total number of times this label was predicted correctly.
    /// In case of <see cref="TestResult.GlobalMetrics"/> it's a total number of
    /// correctly predicted labels (not distinct) through all predictions.
    /// </summary>
    public long PredictedGold { get; set; }

    /// <summary>
    /// An array of all predicted scores for this label versus 1.0 if the prediction
    /// was correct or 0.0 if not. Array will be empty if this is a <see cref="TestResult.GlobalMetrics"/>.
    /// </summary>
    public (float score, float gold)[] ScoreVsTrue { get; set; }

    /// <summary>
    /// Gets precision value. May be <code>double.NaN</code> if <see cref="Predicted"/> is 0.
    /// </summary>
    public double GetPrecision()
    {
        if (Predicted == 0)
            return double.NaN;

        return PredictedGold / (double)Predicted;
    }

    /// <summary>
    /// Gets recall value. May be <code>double.NaN</code> if <see cref="Gold"/> is 0.
    /// </summary>
    public double GetRecall()
    {
        if (Gold == 0)
            return double.NaN;

        return PredictedGold / (double)Gold;
    }

    /// <summary>
    /// Gets F1 value. May be <code>double.NaN</code> if <see cref="Predicted"/> and <see cref="Gold"/> is 0.
    /// </summary>
    public double GetF1()
    {
        if (Predicted + Gold == 0)
            return double.NaN;

        return 2 * PredictedGold / (double)(Predicted + Gold);
    }
}