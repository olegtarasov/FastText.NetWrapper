using System;
using System.IO;
using System.Runtime.InteropServices;
using AutoMapper;

namespace FastText.NetWrapper
{
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
    /// Unsupervised model.
    /// </summary>
    public enum UnsupervisedModel : int
    {
        CBow = 1,
        SkipGram,
    }

    /// <summary>
    /// FastText model.
    /// </summary>
    internal enum ModelName : int
    {
        CBow = 1,
        SkipGram,
        Supervised
    };

    public class QuantizedSupervisedArgs : SupervisedArgs
    {
        public unsafe QuantizedSupervisedArgs()
        {
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultSupervisedArgs(new IntPtr(&argsPtr));
            
            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        }

        internal QuantizedSupervisedArgs(bool dummy)
        {
        }

        /// <summary>
        /// quantizing the classifier [0]
        /// </summary>
        public bool qout {get; set;}
        
        /// <summary>
        /// finetune embeddings if a cutoff is applied [0]
        /// </summary>
        public bool retrain {get; set;}
        
        /// <summary>
        /// quantizing the norm separately [0]
        /// </summary>
        public bool qnorm {get; set;}
        
        /// <summary>
        /// number of words and ngrams to retain [0]
        /// </summary>
        public ulong cutoff {get; set;}
        
        /// <summary>
        /// size of each sub-vector [2]
        /// </summary>
        public ulong dsub {get; set;}
    }

    /// <summary>
    /// Arguments for supervised model training.
    /// </summary>
    public class SupervisedArgs : FastTextArgs
    {
        public unsafe SupervisedArgs()
        {
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultSupervisedArgs(new IntPtr(&argsPtr));
            
            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        } 
    }

    /// <summary>
    /// Arguments for unsupervised learning.
    /// </summary>
    public class UnsupervisedArgs : FastTextArgs
    {
    }
    

    /// <summary>
    /// This class contains all options that can be passed to fastText.
    /// Consult https://github.com/facebookresearch/fastText/blob/master/docs/options.md for their meaning.
    /// </summary>
    public abstract class FastTextArgs
    {
        #region Args

        [DllImport(FastTextWrapper.FastTextDll)]
        protected static extern void GetDefaultArgs(IntPtr args);
        
        [DllImport(FastTextWrapper.FastTextDll)]
        protected static extern void GetDefaultSupervisedArgs(IntPtr args);
        
        [DllImport(FastTextWrapper.FastTextDll)]
        protected static extern void DestroyArgs(IntPtr args);

        #endregion

        protected static readonly IMapper Mapper;

        static FastTextArgs()
        {
            Mapper = new MapperConfiguration(config =>
            {
                config.ShouldMapProperty = prop => prop.GetMethod.IsPublic || prop.GetMethod.IsAssembly;
                config.CreateMap<FastTextWrapper.FastTextArgsStruct, FastTextArgs>();
                config.CreateMap<FastTextWrapper.FastTextArgsStruct, SupervisedArgs>();
                config.CreateMap<FastTextWrapper.FastTextArgsStruct, UnsupervisedArgs>();
                config.CreateMap<FastTextWrapper.FastTextArgsStruct, QuantizedSupervisedArgs>();
            }).CreateMapper();
        }
        
        /// <summary>
        /// This constructor gets values from
        /// https://github.com/olegtarasov/fastText/blob/b0a32d744f4d16d8f9834649f6f178ff79b5a4ce/src/fasttext_api.cc#L12
        /// </summary>
        protected unsafe FastTextArgs()
        {
            LabelPrefix = "__label__";
            
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultArgs(new IntPtr(&argsPtr));

            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        }

        /// <summary>
        /// learning rate [0.1]
        /// </summary>
        public double lr {get; set;}
        
        /// <summary>
        /// change the rate of updates for the learning rate [100]
        /// </summary>
        public int lrUpdateRate {get; set;}
        
        /// <summary>
        /// size of word vectors [100]
        /// </summary>
        public int dim {get; set;}
        
        /// <summary>
        /// size of the context window [5]
        /// </summary>
        public int ws {get; set;}
        
        /// <summary>
        /// number of epochs [5]
        /// </summary>
        public int epoch {get; set;}
        
        /// <summary>
        /// minimal number of word occurrences [1]
        /// </summary>
        public int minCount {get; set;}
        
        /// <summary>
        /// minimal number of label occurrences [0]
        /// </summary>
        public int minCountLabel {get; set;}
        
        /// <summary>
        /// number of negatives sampled [5]
        /// </summary>
        public int neg {get; set;}
        
        /// <summary>
        /// max length of word ngram [1]
        /// </summary>
        public int wordNgrams {get; set;}
        
        /// <summary>
        /// loss function {ns, hs, softmax} [softmax]
        /// </summary>
        public LossName loss {get; set;}
        
        /// <summary>
        /// Model to use.
        /// </summary>
        internal ModelName model {get; set;}
        
        /// <summary>
        /// number of buckets [2000000]
        /// </summary>
        public int bucket {get; set;}
        
        /// <summary>
        /// min length of char ngram [0]
        /// </summary>
        public int minn {get; set;}
        
        /// <summary>
        /// max length of char ngram [0]
        /// </summary>
        public int maxn {get; set;}
        
        /// <summary>
        /// number of threads [12]
        /// </summary>
        public int thread {get; set;}
        
        /// <summary>
        /// sampling threshold [0.0001]
        /// </summary>
        public double t {get; set;}
        public int verbose {get; set;}
        
        /// <summary>
        /// whether output params should be saved [0]
        /// </summary>
        public bool saveOutput {get; set;}
        
        /// <summary>
        /// Fixed random seed.
        /// </summary>
        public int seed { get; set; }

        /// <summary>
        /// labels prefix [__label__]
        /// </summary>
        public string LabelPrefix { get; set; }
        
        /// <summary>
        /// pretrained word vectors for supervised learning []
        /// </summary>
        public string PretrainedVectors { get; set; }
        
        /// <summary>
        /// Optional training progress callback.
        /// </summary>
        public TrainProgressCallback TrainProgressCallback { get; set; }
    }

    /// <summary>
    /// Autotune arguments.
    /// </summary>
    public class AutotuneArgs
    {
        /// <summary>
        /// Default ctor.
        /// </summary>
        public AutotuneArgs()
        {
        }

        /// <summary>
        /// Ctor specifying a validation file.
        /// </summary>
        /// <param name="validationFile">Validation file to tune on.</param>
        public AutotuneArgs(string validationFile)
        {
            ValidationFile = validationFile;
        }

        /// <summary>
        /// Path to a labeled validation file for autotuning. 
        /// </summary>
        public string ValidationFile { get; set; } = "";

        /// <summary>
        /// Metric to autotune with. Default is "f1".
        /// See https://github.com/olegtarasov/fastText/blob/c_api/docs/autotune.md#how-to-set-the-optimization-metric
        /// for possible options.
        /// </summary>
        public string Metric { get; set; } = "f1";

        /// <summary>
        /// Number of predictions to make during autotune. Default is 1.
        /// </summary>
        public int Predictions { get; set; } = 1;

        /// <summary>
        /// Time in seconds to spend on tuning. Default is 60 * 5 = 5 minutes.
        /// </summary>
        public int Duration { get; set; } = 60 * 5;

        /// <summary>
        /// If set, creates a quantized model, also optimizing quantization parameters.
        /// See https://github.com/olegtarasov/fastText/blob/c_api/docs/autotune.md#constrain-model-size
        /// for details.
        /// </summary>
        public string ModelSize { get; set; } = "";

        /// <summary>
        /// Autotune console verbosity. 0 for silent, > 0 for verbose.
        /// </summary>
        public int Verbose { get; set; } = 2;

        /// <summary>
        /// Optional autotune progress callback.
        /// </summary>
        public AutotuneProgressCallback AutotuneProgressCallback { get; set; }
    }

    internal class DebugArgs
    {
        public QuantizedSupervisedArgs ExternalArgs { get; set; } = new QuantizedSupervisedArgs(false);
        public AutotuneArgs ExternalTune { get; set; } = new AutotuneArgs();
        public QuantizedSupervisedArgs ConvertedArgs { get; set; } = new QuantizedSupervisedArgs(false);
        public AutotuneArgs ConvertedTune { get; set; } = new AutotuneArgs();

        public string ExternalInput { get; set; }
        public string ExternalOutput { get; set; }
        public string ConvertedInput { get; set; }
        public string ConvertedOutput { get; set; }
        
        internal static DebugArgs Load(string path)
        {
            var result = new DebugArgs();
            var lines = File.ReadAllLines(path);
            int idx = -1;
            
            if (lines[++idx] != "= eargs")
                throw new InvalidOperationException();

            result.ExternalInput = lines[++idx];
            result.ExternalOutput = lines[++idx];
            
            result.ExternalArgs.lr = double.Parse(lines[++idx]);
            result.ExternalArgs.lrUpdateRate = int.Parse(lines[++idx]);
            result.ExternalArgs.dim = int.Parse(lines[++idx]);
            result.ExternalArgs.ws = int.Parse(lines[++idx]);
            result.ExternalArgs.epoch = int.Parse(lines[++idx]);
            result.ExternalArgs.minCount = int.Parse(lines[++idx]);
            result.ExternalArgs.minCountLabel = int.Parse(lines[++idx]);
            result.ExternalArgs.neg = int.Parse(lines[++idx]);
            result.ExternalArgs.wordNgrams = int.Parse(lines[++idx]);
            result.ExternalArgs.loss = (LossName)int.Parse(lines[++idx]);
            result.ExternalArgs.model = (ModelName)int.Parse(lines[++idx]);
            result.ExternalArgs.bucket = int.Parse(lines[++idx]);
            result.ExternalArgs.minn = int.Parse(lines[++idx]);
            result.ExternalArgs.maxn = int.Parse(lines[++idx]);
            result.ExternalArgs.thread = int.Parse(lines[++idx]);
            result.ExternalArgs.t = double.Parse(lines[++idx]);
            result.ExternalArgs.LabelPrefix = lines[++idx];
            result.ExternalArgs.verbose = int.Parse(lines[++idx]);
            result.ExternalArgs.PretrainedVectors = lines[++idx];
            result.ExternalArgs.saveOutput = int.Parse(lines[++idx]) == 1;
            result.ExternalArgs.seed = int.Parse(lines[++idx]);
            result.ExternalArgs.qout = int.Parse(lines[++idx]) == 1;
            result.ExternalArgs.retrain = int.Parse(lines[++idx]) == 1;
            result.ExternalArgs.qnorm = int.Parse(lines[++idx]) == 1;
            result.ExternalArgs.cutoff = ulong.Parse(lines[++idx]);
            result.ExternalArgs.dsub = ulong.Parse(lines[++idx]);

            result.ExternalTune.ValidationFile = lines[++idx];
            result.ExternalTune.Metric = lines[++idx];
            result.ExternalTune.Predictions = int.Parse(lines[++idx]);
            result.ExternalTune.Duration = int.Parse(lines[++idx]);
            result.ExternalTune.ModelSize = lines[++idx];
            result.ExternalTune.Verbose = int.Parse(lines[++idx]);
            
            if (lines[++idx] != "= args")
                throw new InvalidOperationException();

            result.ConvertedInput = lines[++idx];
            result.ConvertedOutput = lines[++idx];

            result.ConvertedArgs.lr = double.Parse(lines[++idx]);
            result.ConvertedArgs.lrUpdateRate = int.Parse(lines[++idx]);
            result.ConvertedArgs.dim = int.Parse(lines[++idx]);
            result.ConvertedArgs.ws = int.Parse(lines[++idx]);
            result.ConvertedArgs.epoch = int.Parse(lines[++idx]);
            result.ConvertedArgs.minCount = int.Parse(lines[++idx]);
            result.ConvertedArgs.minCountLabel = int.Parse(lines[++idx]);
            result.ConvertedArgs.neg = int.Parse(lines[++idx]);
            result.ConvertedArgs.wordNgrams = int.Parse(lines[++idx]);
            result.ConvertedArgs.loss = (LossName)int.Parse(lines[++idx]);
            result.ConvertedArgs.model = (ModelName)int.Parse(lines[++idx]);
            result.ConvertedArgs.bucket = int.Parse(lines[++idx]);
            result.ConvertedArgs.minn = int.Parse(lines[++idx]);
            result.ConvertedArgs.maxn = int.Parse(lines[++idx]);
            result.ConvertedArgs.thread = int.Parse(lines[++idx]);
            result.ConvertedArgs.t = double.Parse(lines[++idx]);
            result.ConvertedArgs.LabelPrefix = lines[++idx];
            result.ConvertedArgs.verbose = int.Parse(lines[++idx]);
            result.ConvertedArgs.PretrainedVectors = lines[++idx];
            result.ConvertedArgs.saveOutput = int.Parse(lines[++idx]) == 1;
            result.ConvertedArgs.seed = int.Parse(lines[++idx]);
            result.ConvertedArgs.qout = int.Parse(lines[++idx]) == 1;
            result.ConvertedArgs.retrain = int.Parse(lines[++idx]) == 1;
            result.ConvertedArgs.qnorm = int.Parse(lines[++idx]) == 1;
            result.ConvertedArgs.cutoff = ulong.Parse(lines[++idx]);
            result.ConvertedArgs.dsub = ulong.Parse(lines[++idx]);

            result.ConvertedTune.ValidationFile = lines[++idx];
            result.ConvertedTune.Metric = lines[++idx];
            result.ConvertedTune.Predictions = int.Parse(lines[++idx]);
            result.ConvertedTune.Duration = int.Parse(lines[++idx]);
            result.ConvertedTune.ModelSize = lines[++idx];
            result.ConvertedTune.Verbose = int.Parse(lines[++idx]);

            return result;
        }
    }
}