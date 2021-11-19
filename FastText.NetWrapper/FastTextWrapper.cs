using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using AutoMapper;
using Microsoft.Extensions.Logging;

namespace FastText.NetWrapper
{
	/// <summary>
	/// A wrapper around native fastText implementation.
	/// </summary>
	public partial class FastTextWrapper : IDisposable
	{
		private static readonly Encoding _utf8 = Encoding.UTF8;

		private readonly IMapper _mapper;
		private readonly ILogger<FastTextWrapper> _logger;
		
		private IntPtr _fastText;
		private int _maxLabelLen;

		/// <summary>
		/// Ctor.
		/// </summary>
		/// <param name="useBundledLibrary">
		/// If <code>true</code>, a bundled copy of fastText binary is extracted to process' current directory.
		/// You can set this to <code>false</code>, but then you must ensure that a compatible binary for your
		/// platform is discoverable by system library loader.
		/// 
		/// You can compile your own binary from this specific fork: https://github.com/olegtarasov/fastText.
		/// </param>
		/// <param name="loggerFactory">Optional logger factory.</param>
		public FastTextWrapper(ILoggerFactory loggerFactory = null)
		{
			_logger = loggerFactory?.CreateLogger<FastTextWrapper>();
			
			_mapper = new MapperConfiguration(config =>
				{
					config.ShouldMapProperty = prop => prop.GetMethod.IsPublic || prop.GetMethod.IsAssembly;
					config.CreateMap<SupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<QuantizedSupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<UnsupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<AutotuneArgs, AutotuneArgsStruct>();
				})
				.CreateMapper();

			_fastText = CreateFastText();
		}
		
		/// <summary>
		/// Ctor.
		/// </summary>
		/// <param name="useBundledLibrary">
		/// If <code>true</code>, a bundled copy of fastText binary is extracted to process' current directory.
		/// You can set this to <code>false</code>, but then you must ensure that a compatible binary for your
		/// platform is discoverable by system library loader.
		/// 
		/// You can compile your own binary from this specific fork: https://github.com/olegtarasov/fastText.
		/// </param>
		/// <param name="loggerFactory">Optional logger factory.</param>
		[Obsolete("Native libraries are no longer extracted from resources, so there is no need for useBundledLibrary argument.")]
		public FastTextWrapper(bool useBundledLibrary, ILoggerFactory loggerFactory = null)
		{
			_logger = loggerFactory?.CreateLogger<FastTextWrapper>();
			
			_mapper = new MapperConfiguration(config =>
				{
					config.ShouldMapProperty = prop => prop.GetMethod.IsPublic || prop.GetMethod.IsAssembly;
					config.CreateMap<SupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<QuantizedSupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<UnsupervisedArgs, FastTextArgsStruct>();
					config.CreateMap<AutotuneArgs, AutotuneArgsStruct>();
				})
				.CreateMapper();

			_fastText = CreateFastText();
		}

		/// <summary>
		/// Path to a model binary. Can be empty if model is not trained or loaded, or
		/// if model was loaded from memory.
		/// </summary>
		public string ModelPath { get; private set; } = string.Empty;

		#region Model management

		/// <summary>
		/// Loads a trained model from a file.
		/// </summary>
		/// <param name="path">Path to a model (.bin file).</param>
		public void LoadModel(string path)
		{
			CheckForErrors(LoadModel(_fastText, path));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
			ModelPath = path;
		}

		/// <summary>
		/// Loads a trained model from a byte array.
		/// </summary>
		/// <param name="bytes">Bytes array containing the model (.bin file).</param>
		public void LoadModel(byte[] bytes)
		{
			CheckForErrors(LoadModelData(_fastText, bytes, bytes.Length));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
			ModelPath = string.Empty;
		}
		
		#endregion

		#region Model info

		/// <summary>
		/// Gets all labels that classifier was trained on.
		/// </summary>
		/// <returns>Labels.</returns>
		public unsafe string[] GetLabels()
		{
			CheckModelLoaded();
			
			IntPtr labelsPtr;
			int numLabels = CheckForErrors(GetLabels(_fastText, new IntPtr(&labelsPtr)));

			var result = new string[numLabels];
			for (int i = 0; i < numLabels; i++)
			{
				var ptr = Marshal.ReadIntPtr(labelsPtr, i * IntPtr.Size);
				result[i] = _utf8.GetString(GetStringBytes(ptr));
			}

			DestroyStrings(labelsPtr, numLabels);

			return result;
		}

		/// <summary>
		/// Returns <code>true</code> if a model was trained or loaded and is ready.
		/// </summary>
		public bool IsModelReady()
		{
			return IsModelReady(_fastText);
		}
		
		/// <summary>
		/// Gets the vector dimension of the loaded model.
		/// </summary>
		public int GetModelDimension()
		{
			return GetModelDimension(_fastText);
		}
		
		#endregion

		#region FastText commands

		/// <summary>
		/// Trains a new supervised model.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Supervised(string inputPath, string outputPath)
		{
			Supervised(inputPath, outputPath, new SupervisedArgs(), new AutotuneArgs(), false);
		}

		/// <summary>
		/// Trains a new supervised model.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">
		/// Training arguments. If <see cref="SupervisedArgs"/> is passed, a supervised model will be trained.
		/// If <see cref="QuantizedSupervisedArgs"/> is passed, model will be quantized after training.
		/// </param>
		/// <param name="progressCallback">Optional progress callback.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Supervised(string inputPath, string outputPath, SupervisedArgs args, TrainProgressCallback progressCallback = null)
		{
			Supervised(inputPath, outputPath, args, new AutotuneArgs(), false);
		}

		/// <summary>
		/// Trains a new supervised model. If <see cref="AutotuneArgs.ValidationFile"/> is specified, an automated
		/// hyperparameter search will be performed.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">
		/// Training arguments. If <see cref="SupervisedArgs"/> is passed, a supervised model will be trained.
		/// If <see cref="QuantizedSupervisedArgs"/> is passed, model will be quantized after training.
		/// </param>
		/// <param name="autotuneArgs">Autotune arguments.</param>
		/// <param name="progressCallback">Optional progress callback.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Supervised(string inputPath, string outputPath, SupervisedArgs args, AutotuneArgs autotuneArgs, TrainProgressCallback progressCallback = null)
		{
			Supervised(inputPath, outputPath, args, autotuneArgs, false);
		}

		/// <summary>
		/// Trains a new supervised model. If <see cref="AutotuneArgs.ValidationFile"/> is specified, an automated
		/// hyperparameter search will be performed.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">
		/// Training arguments. If <see cref="SupervisedArgs"/> is passed, a supervised model will be trained.
		/// If <see cref="QuantizedSupervisedArgs"/> is passed, model will be quantized after training.
		/// </param>
		/// <param name="autotuneArgs">Autotune arguments.</param>
		/// <param name="debug">Whether to write debug info.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		internal void Supervised(string inputPath, string outputPath, SupervisedArgs args, AutotuneArgs autotuneArgs, bool debug)
		{
			ValidatePaths(inputPath, outputPath, args.PretrainedVectors);

			if (args.model != ModelName.Supervised)
			{
				_logger?.LogWarning($"{args.model} model type specified in a Supervised() call. Model type will be changed to Supervised.");
			}

			var quantizedArgs = args as QuantizedSupervisedArgs;
			if (!string.IsNullOrEmpty(autotuneArgs.ModelSize) && quantizedArgs == null)
			{
				throw new InvalidOperationException("You specified model size in autotuneArgs, but passed SupervisedArgs instance. Pass QuantizedSupervisedArgs instead.");
			}

			bool quantizeWithNoQuantTune = quantizedArgs != null && string.IsNullOrEmpty(autotuneArgs.ModelSize);

			var argsStruct = _mapper.Map<FastTextArgsStruct>(args);
			argsStruct.model = model_name.sup;

			var autotuneStruct = _mapper.Map<AutotuneArgsStruct>(autotuneArgs);
			CheckForErrors(Train(
				_fastText, 
				inputPath, 
				quantizeWithNoQuantTune ? null : outputPath, 
				argsStruct, 
				autotuneStruct,
				args.TrainProgressCallback,
				autotuneArgs.AutotuneProgressCallback,
				args.LabelPrefix, 
				args.PretrainedVectors, 
				debug));

			if (quantizeWithNoQuantTune)
			{
				Quantize(quantizedArgs, outputPath);
			}
			else
			{
				_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
				ModelPath = AdjustPath(outputPath, !string.IsNullOrEmpty(autotuneArgs.ModelSize));
			}
		}

		/// <summary>
		/// Trains a new unsupervised model.
		/// </summary>
		/// <param name="model">Type of unsupervised model: Skipgram or Cbow.</param>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Unsupervised(UnsupervisedModel model, string inputPath, string outputPath)
		{
			Unsupervised(model, inputPath, outputPath, new UnsupervisedArgs());
		}

		/// <summary>
		/// Trains a new unsupervised model.
		/// </summary>
		/// <param name="model">Type of unsupervised model: Skipgram or Cbow.</param>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Low-level training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Unsupervised(UnsupervisedModel model, string inputPath, string outputPath, UnsupervisedArgs args)
		{
			ValidatePaths(inputPath, outputPath, args.PretrainedVectors);

			args.model = (ModelName)model;
			
			var argsStruct = _mapper.Map<FastTextArgsStruct>(args);
			CheckForErrors(Train(
				_fastText, 
				inputPath, 
				outputPath, 
				argsStruct, 
				new AutotuneArgsStruct(), 
				args.TrainProgressCallback, 
				null, 
				args.LabelPrefix, 
				args.PretrainedVectors, 
				false));
			_maxLabelLen = 0;

			ModelPath = AdjustPath(outputPath, false);
		}

		public void Quantize(string output = null) => Quantize(new QuantizedSupervisedArgs(), output);

		/// <summary>
		/// Quantize a loaded model.
		/// </summary>
		/// <param name="args">Quantization args.</param>
		/// <param name="output">Custom output path. Required if model was loaded from memory.</param>
		public void Quantize(QuantizedSupervisedArgs args, string output = null)
		{
			if (!IsModelReady())
				throw new InvalidOperationException("Model is not loaded or trained!");
			
			if (string.IsNullOrEmpty(ModelPath) && string.IsNullOrEmpty(output))
				throw new InvalidOperationException("Model was loaded from memory. You need to specify output path.");

			var argsStruct = _mapper.Map<FastTextArgsStruct>(args);
			string outPath = AdjustPath(string.IsNullOrEmpty(output) ? ModelPath : output, true);
			
			if ((Path.IsPathRooted(output) && !Directory.Exists(Path.GetDirectoryName(outPath))))
				throw new InvalidOperationException("Output directory doesn't exist!");

			CheckForErrors(Quantize(_fastText, outPath, argsStruct, args.LabelPrefix));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));

			ModelPath = outPath;
		}

		/// <summary>
		/// Calculate nearest neighbors from input text.
		/// </summary>
		/// <param name="text">Text to calculate nearest neighbors from.</param>
		/// <param name="number">Number of neighbors.</param>
		/// <returns>Nearest neighbor predictions.</returns>
		[Obsolete("Please use GetNearestNeighbours method")]
		public Prediction[] GetNN(string text, int number) => GetNearestNeighbours(text, number);
		
		
		/// <summary>
		/// Calculate nearest neighbors from input text.
		/// </summary>
		/// <param name="text">Text to calculate nearest neighbors from.</param>
		/// <param name="number">Number of neighbors.</param>
		/// <returns>Nearest neighbor predictions.</returns>
		public unsafe Prediction[] GetNearestNeighbours(string text, int number)
		{
			CheckModelLoaded();

			var probs = new float[number];
			IntPtr labelsPtr;

			int cnt = CheckForErrors(GetNN(_fastText, _utf8.GetBytes(text), new IntPtr(&labelsPtr), probs, number));
			var result = new Prediction[cnt];

			for (int i = 0; i < cnt; i++)
			{
				var ptr = Marshal.ReadIntPtr(labelsPtr, i * IntPtr.Size);
				string label = _utf8.GetString(GetStringBytes(ptr));
				result[i] = new Prediction(probs[i], label);
			}

			DestroyStrings(labelsPtr, cnt);

			return result;
		}

		/// <summary>
		/// Vectorizes a sentence.
		/// </summary>
		/// <param name="text">Sentence to vectorize.</param>
		/// <returns>A single averaged vector.</returns>
		public unsafe float[] GetSentenceVector(string text)
		{
			CheckModelLoaded();
			
			IntPtr vecPtr;
			int dim = CheckForErrors(GetSentenceVector(_fastText, _utf8.GetBytes(text), new IntPtr(&vecPtr)));

			var result = new float[dim];
			long sz = sizeof(float) * dim;
			fixed (void* resPtr = &result[0])
			{
				Buffer.MemoryCopy(vecPtr.ToPointer(), resPtr, sz, sz);
			}

			DestroyVector(vecPtr);

			return result;
		}

		/// <summary>
		/// Vectorizes a word.
		/// </summary>
		/// <param name="word">Word to vectorize.</param>
		/// <returns>A single vector.</returns>
		public unsafe float[] GetWordVector(string word)
		{
			CheckModelLoaded();
			
			IntPtr vecPtr;
			int dim = CheckForErrors(GetWordVector(_fastText, _utf8.GetBytes(word), new IntPtr(&vecPtr)));

			var result = new float[dim];
			long sz = sizeof(float) * dim;
			fixed (void* resPtr = &result[0])
			{
				Buffer.MemoryCopy(vecPtr.ToPointer(), resPtr, sz, sz);
			}

			DestroyVector(vecPtr);

			return result;
		}

		#endregion

		#region Predictions

		/// <summary>
		/// Predicts a single label from input text.
		/// </summary>
		/// <param name="text">Text to predict a label from.</param>
		/// <returns>Single prediction.</returns>
		public unsafe Prediction PredictSingle(string text)
		{
			CheckModelLoaded();
			CheckModelLabels();

			IntPtr labelPtr;
			float prob = CheckForErrors(PredictSingle(_fastText, _utf8.GetBytes(text), new IntPtr(&labelPtr)));

			string label = _utf8.GetString(GetStringBytes(labelPtr));
			DestroyString(labelPtr);

			return new Prediction(prob, label);
		}

		/// <summary>
		/// Predicts multiple labels from input text.
		/// </summary>
		/// <param name="text">Text to predict labels from.</param>
		/// <param name="number">Number of labels to predict.</param>
		/// <returns>Multiple predictions.</returns>
		public unsafe Prediction[] PredictMultiple(string text, int number)
		{
			CheckModelLoaded();
			CheckModelLabels();

			var probs = new float[number];
			IntPtr labelsPtr;

			int cnt = CheckForErrors(PredictMultiple(_fastText, _utf8.GetBytes(text), new IntPtr(&labelsPtr), probs, number));
			var result = new Prediction[cnt];

			for (int i = 0; i < cnt; i++)
			{
				var ptr = Marshal.ReadIntPtr(labelsPtr, i * IntPtr.Size);
				string label = _utf8.GetString(GetStringBytes(ptr));
				result[i] = new Prediction(probs[i], label);
			}

			DestroyStrings(labelsPtr, cnt);

			return result;
		}

		#endregion

		#region Testing

		public TestResult Test(string inputPath, int k = 1, float threshold = 0.0f)
		{
			return TestInternal(inputPath, k, threshold, false);
		}

		internal unsafe TestResult TestInternal(string inputPath, int k, float threshold, bool debug)
		{
			IntPtr meterPtr;
			CheckForErrors(Test(_fastText, inputPath, k, threshold, new IntPtr(&meterPtr), debug));

			var labels = GetLabels();
			var meter = Marshal.PtrToStructure<TestMeter>(meterPtr);
			var globalMetrics = new Metrics(Marshal.PtrToStructure<TestMetrics>(meter.Metrics), null);
			var srcLabelMetrics = new Metrics[meter.Labels];

			for (int i = 0; i < meter.Labels; i++)
			{
				var ptr = Marshal.ReadIntPtr(meter.LabelMetrics, i * IntPtr.Size);
				var testMetrics = Marshal.PtrToStructure<TestMetrics>(ptr);
				srcLabelMetrics[i] = new Metrics(testMetrics, labels[testMetrics.Label]);
			}
			
			var result = new TestResult(meter.Examples, globalMetrics, srcLabelMetrics);

			DestroyMeter(meterPtr);

			return result;
		}

		#endregion
		
		/// <inheritdoc />
		public void Dispose()
		{
			if (_fastText == IntPtr.Zero)
			{
				return;
			}

			DestroyFastText(_fastText);
			_fastText = IntPtr.Zero;
		}

		private string AdjustPath(string path, bool isQuantized)
		{
			string result = Path.HasExtension(path) ? Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(path)) : path;
			result += isQuantized ? ".ftz" : ".bin";

			return result;
		}

		private int CheckForErrors(int result)
		{
			if (result != -1)
			{
				return result;
			}
			
			ThrowNativeException();

			return -1;
		}
		
		private float CheckForErrors(float result)
		{
			if (Math.Abs(result - (-1)) > 10e-5)
			{
				return result;
			}
			
			ThrowNativeException();

			return -1;
		}

		private unsafe void ThrowNativeException()
		{
			IntPtr errorPtr;
			GetLastErrorText(new IntPtr(&errorPtr));

			string error = _utf8.GetString(GetStringBytes(errorPtr));
			
			DestroyString(errorPtr);
			
			throw new NativeLibraryException(error);
		}

		private void CheckModelLabels()
		{
			if (_maxLabelLen == 0)
			{
				throw new InvalidOperationException("Loaded model doesn't contain supervised labels. Maybe you loaded an unsupervised model?");
			}
		}
		
		private void CheckModelLoaded()
		{
			if (!IsModelReady())
			{
				throw new InvalidOperationException("Model not ready! Train a new model or load an existing one.");
			}
		}
		
		private void ValidatePaths(string input, string output, string pretrained)
		{
			if (string.IsNullOrEmpty(input) || !File.Exists(input))
			{
				throw new FileNotFoundException($"Invalid input file name!", input);
			}

			if (string.IsNullOrEmpty(output) || (Path.IsPathRooted(output) && !Directory.Exists(Path.GetDirectoryName(output))))
			{
				throw new DirectoryNotFoundException("Invalid output directory!");
			}

			if (pretrained != null && !File.Exists(pretrained))
			{
				throw new FileNotFoundException("Invalid pretrained vectors path!", pretrained);
			}
		}

		private unsafe byte[] GetStringBytes(IntPtr ptr)
		{
			var bPtr = (byte*)ptr.ToPointer();
			int len = 0;

			while (*bPtr != 0)
			{
				len++;
				bPtr++;
			}

			if (len == 0)
			{
				return Array.Empty<byte>();
			}

			var result = new byte[len];
			Marshal.Copy(ptr, result, 0, len);

			return result;
		}
	}
}
