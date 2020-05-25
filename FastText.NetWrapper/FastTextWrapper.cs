using System;
using System.IO;
using System.Net.Http.Headers;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using AutoMapper;
using Microsoft.Extensions.Logging;
using NativeLibraryManager;

namespace FastText.NetWrapper
{
	/// <summary>
	/// A wrapper around native fastText implementation.
	/// </summary>
	public partial class FastTextWrapper : IDisposable
	{
		private static readonly Encoding _utf8 = Encoding.UTF8;

		private readonly IMapper _mapper;
		
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
		public FastTextWrapper(bool useBundledLibrary = true, ILoggerFactory loggerFactory = null)
		{
			if (useBundledLibrary)
			{
				var accessor = new ResourceAccessor(Assembly.GetExecutingAssembly());
				var manager = new LibraryManager(
					loggerFactory,
					new LibraryItem(Platform.Windows, Bitness.x64,
						new LibraryFile("fasttext.dll", accessor.Binary("Resources.fasttext.dll"))),
					new LibraryItem(Platform.MacOs, Bitness.x64,
						new LibraryFile("libfasttext.dylib", accessor.Binary("Resources.libfasttext.dylib"))),
					new LibraryItem(Platform.Linux, Bitness.x64,
						new LibraryFile("libfasttext.so", accessor.Binary("Resources.libfasttext.so"))));

				manager.LoadNativeLibrary();
			}
			
			_mapper = new MapperConfiguration(config => config.CreateMap<FastTextArgs, FastTextArgsStruct>())
				.CreateMapper();

			_fastText = CreateFastText();
		}

		#region Model management

		/// <summary>
		/// Loads a trained model from a file.
		/// </summary>
		/// <param name="path">Path to a model (.bin file).</param>
		public void LoadModel(string path)
		{
			CheckForErrors(LoadModel(_fastText, path));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
		}

		/// <summary>
		/// Loads a trained model from a byte array.
		/// </summary>
		/// <param name="bytes">Bytes array containing the model (.bin file).</param>
		public void LoadModel(byte[] bytes)
		{
			CheckForErrors(LoadModelData(_fastText, bytes, bytes.Length));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
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
		/// Use <see cref="FastTextArgs.SupervisedDefaults"/> to get reasonable default args for
		/// supervised training.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Low-level training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Supervised(string inputPath, string outputPath, FastTextArgs args)
		{
			ValidatePaths(inputPath, outputPath, args.PretrainedVectors);

			var argsStruct = _mapper.Map<FastTextArgsStruct>(args);
			argsStruct.model = model_name.sup;
			CheckForErrors(Supervised(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix, args.PretrainedVectors));
			_maxLabelLen = CheckForErrors(GetMaxLabelLength(_fastText));
		}

		/// <summary>
		/// Calculate nearest neighbors from input text.
		/// </summary>
		/// <param name="text">Text to calculate nearest neighbors from.</param>
		/// <param name="number">Number of neighbors.</param>
		/// <returns>Nearest neighbor predictions.</returns>
		public unsafe Prediction[] GetNN(string text, int number)
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

		#region Deprecated

		/// <summary>
		/// Trains a new supervised classification model.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		[Obsolete("This method is deprecated and will be removed in v. 1.1. Use `Train` overload with `FastTextArgs`.")]
		public void Train(string inputPath, string outputPath, SupervisedArgs args)
		{
			ValidatePaths(inputPath, outputPath, null);

			var argsStruct = new SupervisedArgsStruct
							{
								Epochs = args.Epochs,
								LearningRate = args.LearningRate,
								MaxCharNGrams = args.MaxCharNGrams,
								MinCharNGrams = args.MinCharNGrams,
								Verbose = args.Verbose,
								WordNGrams = args.WordNGrams,
								Threads = args.Threads ?? 0
							};

			CheckForErrors(TrainSupervised(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix));
			_maxLabelLen = GetMaxLabelLength(_fastText);
		}

		/// <summary>
		/// Trains a new model using low-level FastText arguments.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Low-level training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		[Obsolete("This method is obsolete. Use one of the new methods: Supervised")]
		public void Train(string inputPath, string outputPath, FastTextArgs args)
		{
			ValidatePaths(inputPath, outputPath, args.PretrainedVectors);

			var argsStruct = _mapper.Map<FastTextArgsStruct>(args);
			CheckForErrors(Train(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix, args.PretrainedVectors));
			_maxLabelLen = GetMaxLabelLength(_fastText);
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

			if (string.IsNullOrEmpty(output) || !Directory.Exists(Path.GetDirectoryName(output)))
			{
				throw new DirectoryNotFoundException("Invalid output directory!");
			}

			if (pretrained != null && (!File.Exists(pretrained)))
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
