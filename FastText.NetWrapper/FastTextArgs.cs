namespace FastText.NetWrapper
{
    /// <summary>
    /// FastText model.
    /// </summary>
    public enum ModelName : int
    {
        CBow = 1,
        SkipGram,
        Supervised
    };

    /// <summary>
    /// Training loss.
    /// </summary>
    public enum LossName : int
    {
        HierarchicalSoftmax = 1,
        NegativeSampling,
        Softmax,
        OneVsAll
    };

    /// <summary>
    /// This class contains all options that can be passed to fastText.
    /// Consult https://github.com/facebookresearch/fastText/blob/master/docs/options.md for their meaning.
    /// </summary>
    public class FastTextArgs
    {
        /// <summary>
        /// This constructor mirrors the default values from
        /// https://github.com/facebookresearch/fastText/blob/0a5759475265705b485fa9fae4d1186d248049aa/src/args.cc#L18
        /// </summary>
        public FastTextArgs()
        {
            lr = 0.05;
            dim = 100;
            ws = 5;
            epoch = 5;
            minCount = 5;
            minCountLabel = 0;
            neg = 5;
            wordNgrams = 1;
            loss = LossName.NegativeSampling;
            model = ModelName.SkipGram;
            bucket = 2000000;
            minn = 3;
            maxn = 6;
            thread = 12;
            lrUpdateRate = 100;
            t = 1e-4;
            verbose = 2;
            saveOutput = false;

            qout = false;
            retrain = false;
            qnorm = false;
            cutoff = 0;
            dsub = 2;

            LabelPrefix = "__label__";
        }

        public double lr {get; set;}
        public int lrUpdateRate {get; set;}
        public int dim {get; set;}
        public int ws {get; set;}
        public int epoch {get; set;}
        public int minCount {get; set;}
        public int minCountLabel {get; set;}
        public int neg {get; set;}
        public int wordNgrams {get; set;}
        public LossName loss {get; set;}
        public ModelName model {get; set;}
        public int bucket {get; set;}
        public int minn {get; set;}
        public int maxn {get; set;}
        public int thread {get; set;}
        public double t {get; set;}
        public int verbose {get; set;}
        public bool saveOutput {get; set;}
        public bool qout {get; set;}
        public bool retrain {get; set;}
        public bool qnorm {get; set;}
        public ulong cutoff {get; set;}
        public ulong dsub {get; set;}

        public string LabelPrefix { get; set; }
        public string PretrainedVectors { get; set; }
    }
}