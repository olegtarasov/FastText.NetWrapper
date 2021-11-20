using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FastText.NetWrapper;

public class TestResult
{
    /// <summary>
    /// Ctor.
    /// </summary>
    public TestResult()
    {
    }

    internal TestResult(long examples, Metrics globalMetrics, Metrics[] labelMetrics)
    {
        GlobalMetrics = globalMetrics;
        LabelMetrics = labelMetrics.ToDictionary(x => x.Label, x => x);
        Examples = examples;
    }

    /// <summary>
    /// Total number of examples in a test.
    /// </summary>
    public long Examples { get; set; }

    /// <summary>
    /// Aggregated metrics for the test run.
    /// </summary>
    public Metrics GlobalMetrics { get; set; }

    /// <summary>
    /// Per-label metrics for the test run.
    /// </summary>
    public Dictionary<string, Metrics> LabelMetrics { get; set; }

    /// <summary>
    /// Gets an array <see cref="Metrics.ScoreVsTrue"/> for specified label, sorted by score.
    /// If <see cref="label"/> is null, this method returns aggregated and sorted array through
    /// all the labels.
    /// </summary>
    /// <param name="label">Valid label or <code>null</code>.</param>
    /// <exception cref="InvalidOperationException">When label was not found in test results.</exception>
    public (float score, float gold)[] GetSortedScoreVsTrue(string label = null)
    {
        if (label != null && !LabelMetrics.ContainsKey(label))
            throw new InvalidOperationException($"Label {label} not found in test results!");
            
        var result = new List<(float, float)>();
        if (label == null)
        {
            foreach (var metrics in LabelMetrics.Values)
                result.AddRange(metrics.ScoreVsTrue);
        }
        else
            result.AddRange(LabelMetrics[label].ScoreVsTrue);

        return result.OrderBy(x => x.Item1).ThenBy(x => x.Item2).ToArray();
    }

    /// <summary>
    /// Gets accumulated true and false positive counts for a specified label (based on <see cref="Metrics.ScoreVsTrue"/>).
    /// If <see cref="label"/> is null, this method returns aggregated and sorted array through
    /// all the labels.
    /// </summary>
    /// <param name="label">Valid label or <code>null</code>.</param>
    /// <exception cref="InvalidOperationException">When label was not found in test results.</exception>
    public (long truePositives, long falsePositives)[] GetPositiveCounts(string label = null)
    {
        var scores = GetSortedScoreVsTrue(label);
        long truePositives = 0;
        long falsePositives = 0;
        var lastScore = -2.0;
        var result = new List<(long, long)>();

        for (var i = scores.Length - 1; i >= 0; i--)
        {
            double score = scores[i].score;
            double gold = scores[i].gold;

            if (score < 0) // only reachable recall
                break;

            if (gold == 1.0)
                truePositives++;
            else
                falsePositives++;

            if (score == lastScore && result.Count > 0) // squeeze tied scores
                result[result.Count - 1] = (truePositives, falsePositives);
            else
                result.Add((truePositives, falsePositives));

            lastScore = score;
        }

        return result.ToArray();
    }

    /// <summary>
    /// Gets a precision-recall curve for a specified label.
    /// If <see cref="label"/> is null, this method returns aggregated result through
    /// all the labels.
    /// </summary>
    /// <param name="label">Valid label or <code>null</code>.</param>
    /// <exception cref="InvalidOperationException">When label was not found in test results.</exception>
    public (double precision, double recall)[] GetPrecisionRecallCurve(string label = null)
    {
        var positiveCounts = GetPositiveCounts(label);

        if (positiveCounts.Length == 0)
            return Array.Empty<(double, double)>();
            
        var result = new List<(double, double)>();
        long golds = label == null ? GlobalMetrics.Gold : LabelMetrics[label].Gold;
        int fullRecallIdx = Array.FindIndex(positiveCounts, item => item.truePositives >= golds);
        fullRecallIdx = fullRecallIdx == -1 ? positiveCounts.Length : fullRecallIdx + 1;

        for (int i = 0; i < fullRecallIdx; i++)
        {
            double precision = 0.0d;
            double truePositives = positiveCounts[i].truePositives;
            double falsePositives = positiveCounts[i].falsePositives;

            if (truePositives + falsePositives != 0)
                precision = truePositives / (truePositives + falsePositives);

            double recall = golds != 0 ? (truePositives / (double)golds) : double.NaN;
            result.Add((precision, recall));
        }
            
        result.Add((1.0, 0.0));

        return result.ToArray();
    }

    /// <summary>
    /// Gets precision at specified recall value for a specified label.
    /// If <see cref="label"/> is null, this method returns aggregated result through
    /// all the labels.
    /// </summary>
    /// <param name="recall">Recall to get precision at.</param>
    /// <param name="label">Valid label or <code>null</code>.</param>
    /// <exception cref="InvalidOperationException">When label was not found in test results.</exception>
    public double GetPrecisionAtRecall(double recall, string label = null)
    {
        var curve = GetPrecisionRecallCurve(label);
        double bestPrecision = 0.0d;

        for (int i = 0; i < curve.Length; i++)
        {
            if (curve[i].recall >= recall)
                bestPrecision = Math.Max(bestPrecision, curve[i].precision);
        }

        return bestPrecision;
    }

    /// <summary>
    /// Gets precision at specified recall value for a specified label.
    /// If <see cref="label"/> is null, this method returns aggregated result through
    /// all the labels.
    /// </summary>
    /// <param name="precision">Recall to get precision at.</param>
    /// <param name="label">Valid label or <code>null</code>.</param>
    /// <exception cref="InvalidOperationException">When label was not found in test results.</exception>
    public double GetRecallAtPrecision(double precision, string label = null)
    {
        var curve = GetPrecisionRecallCurve(label);
        double bestRecall = 0.0d;

        for (int i = 0; i < curve.Length; i++)
        {
            if (curve[i].precision >= precision)
                bestRecall = Math.Max(bestRecall, curve[i].recall);
        }

        return bestRecall;
    }

    internal static (TestResult result, (double precision, double recall)[] curve) LoadDebugResult(string path, string[] labels)
    {
        var result = new TestResult {GlobalMetrics = new Metrics(), LabelMetrics = new Dictionary<string, Metrics>()};
        var curve = new List<(double, double)>();
            
        var lines = File.ReadAllLines(path);
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (line == "= g")
            {
                result.Examples = long.Parse(lines[++i]);
            }
            else if (line.StartsWith("= c"))
            {
                int cnt = int.Parse(lines[++i]);
                for (int j = 0; j < cnt; j++)
                {
                    var parts = lines[++i].Split(';');
                    curve.Add((double.Parse(parts[0]), double.Parse(parts[1])));
                }
            }
            else if (line.StartsWith("="))
            {
                int labelIdx = int.Parse(line.Substring(2));
                Metrics metrics;

                if (labelIdx == -1)
                    metrics = result.GlobalMetrics;
                else
                {
                    metrics = new Metrics {Label = labels[labelIdx]};
                    result.LabelMetrics[labels[labelIdx]] = metrics;
                }

                metrics.Gold = long.Parse(lines[++i]);
                metrics.Predicted = long.Parse(lines[++i]);
                metrics.PredictedGold = long.Parse(lines[++i]);

                int cnt = int.Parse(lines[++i]);
                metrics.ScoreVsTrue = new (float score, float gold)[cnt];

                for (int j = 0; j < cnt; j++)
                {
                    var parts = lines[++i].Split(';');
                    metrics.ScoreVsTrue[j] = (float.Parse(parts[0]), float.Parse(parts[1]));
                }
            }
            else
                continue;
        }

        return (result, curve.ToArray());
    }
}