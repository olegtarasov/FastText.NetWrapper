using System;

namespace FastText.NetWrapper
{
    public class TestResult
    {
        internal TestResult(FastTextWrapper.TestMeter meter, Metrics globalMetrics, Metrics[] labelMetrics)
        {
            GlobalMetrics = globalMetrics;
            LabelMetrics = labelMetrics;
            Examples = meter.Examples;
        }

        public long Examples { get; set; }

        public Metrics GlobalMetrics { get; set; }
        public Metrics[] LabelMetrics { get; set; }
    }
}