using System;
using System.Runtime.InteropServices;
using AutoMapper;

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

    public class SupervisedArgs : FastTextArgs
    {
        public unsafe SupervisedArgs() : base(false)
        {
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultSupervisedArgs(new IntPtr(&argsPtr));
            
            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        }
    }
    

    /// <summary>
    /// This class contains all options that can be passed to fastText.
    /// Consult https://github.com/facebookresearch/fastText/blob/master/docs/options.md for their meaning.
    /// </summary>
    public class FastTextArgs
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
            }).CreateMapper();
        }
        
        /// <summary>
        /// This constructor gets values from
        /// https://github.com/olegtarasov/fastText/blob/b0a32d744f4d16d8f9834649f6f178ff79b5a4ce/src/fasttext_api.cc#L12
        /// </summary>
        internal unsafe FastTextArgs() : this(false)
        {
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultArgs(new IntPtr(&argsPtr));

            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        }

        protected FastTextArgs(bool dummy)
        {
            LabelPrefix = "__label__";
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

        /// <summary>
        /// labels prefix [__label__]
        /// </summary>
        public string LabelPrefix { get; set; }
        
        /// <summary>
        /// pretrained word vectors for supervised learning []
        /// </summary>
        public string PretrainedVectors { get; set; }
    }
}