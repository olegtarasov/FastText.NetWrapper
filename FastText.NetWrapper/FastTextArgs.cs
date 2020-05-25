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

    /// <summary>
    /// This class contains all options that can be passed to fastText.
    /// Consult https://github.com/facebookresearch/fastText/blob/master/docs/options.md for their meaning.
    /// </summary>
    public class FastTextArgs
    {
        #region Args

        [DllImport(FastTextWrapper.FastTextDll)]
        private static extern void GetDefaultArgs(IntPtr args);
        
        [DllImport(FastTextWrapper.FastTextDll)]
        private static extern void GetDefaultSupervisedArgs(IntPtr args);
        
        [DllImport(FastTextWrapper.FastTextDll)]
        private static extern void DestroyArgs(IntPtr args);

        #endregion

        private static readonly IMapper Mapper;

        static FastTextArgs()
        {
            Mapper = new MapperConfiguration(config =>
            {
                config.CreateMap<FastTextWrapper.FastTextArgsStruct, FastTextArgs>();
            }).CreateMapper();
        }
        
        /// <summary>
        /// This constructor gets values from
        /// https://github.com/olegtarasov/fastText/blob/b0a32d744f4d16d8f9834649f6f178ff79b5a4ce/src/fasttext_api.cc#L12
        /// </summary>
        public unsafe FastTextArgs() : this(false)
        {
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultArgs(new IntPtr(&argsPtr));

            Mapper.Map(*argsPtr, this);
            
            DestroyArgs(new IntPtr(argsPtr));
        }

        private FastTextArgs(bool dummy)
        {
            LabelPrefix = "__label__";
        }

        /// <summary>
        /// Returns the same supervised args defaults as in
        /// https://github.com/olegtarasov/fastText/blob/b0a32d744f4d16d8f9834649f6f178ff79b5a4ce/src/fasttext_api.cc#L41
        /// </summary>
        /// <returns></returns>
        public static unsafe FastTextArgs SupervisedDefaults()
        {
            var result = new FastTextArgs(false);
            
            FastTextWrapper.FastTextArgsStruct* argsPtr;

            GetDefaultSupervisedArgs(new IntPtr(&argsPtr));

            Mapper.Map(*argsPtr, result);
            
            DestroyArgs(new IntPtr(argsPtr));

            return result;
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
        public int seed { get; set; }
        public bool qout {get; set;}
        public bool retrain {get; set;}
        public bool qnorm {get; set;}
        public ulong cutoff {get; set;}
        public ulong dsub {get; set;}

        public string LabelPrefix { get; set; }
        public string PretrainedVectors { get; set; }
    }
}