using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using FastText.NetWrapper.Logging;
using NativeLibraryManager;

namespace FastText.NetWrapper
{
	/// <summary>
	/// A wrapper around native fastText implementation.
	/// </summary>
	public partial class FastTextWrapper : IDisposable
	{
		private static readonly ILog _log = LogProvider.For<FastTextWrapper>();
		private static readonly Encoding _utf8 = Encoding.UTF8;

		private IntPtr _fastText;
		private bool _modelLoaded = false;
		private int _maxLabelLen;

		/// <summary>
		/// Ctor.
		/// </summary>
		public FastTextWrapper()
		{
			var accessor = new ResourceAccessor(Assembly.GetExecutingAssembly());
			var manager = new LibraryManager(
				Assembly.GetExecutingAssembly(),
				new LibraryItem(Platform.Windows, Bitness.x64,
					new LibraryFile("fasttext.dll", accessor.Binary("Resources.fasttext.dll"))),
				new LibraryItem(Platform.MacOs, Bitness.x64,
					new LibraryFile("libfasttext.dylib", accessor.Binary("Resources.libfasttext.dylib"))),
				new LibraryItem(Platform.Linux, Bitness.x64,
					new LibraryFile("libfasttext.so", accessor.Binary("Resources.libfasttext.so"))));
			
			manager.LoadNativeLibrary();
			
			_fastText = CreateFastText();
		}

		/// <summary>
		/// Loads a trained model from a file.
		/// </summary>
		/// <param name="path">Path to a model (.bin file).</param>
		public void LoadModel(string path)
		{
			LoadModel(_fastText, path);
			_maxLabelLen = GetMaxLabelLenght(_fastText);
			_modelLoaded = true;
		}

		/// <summary>
		/// Gets all labels that classifier was trained on.
		/// </summary>
		/// <returns>Labels.</returns>
		public unsafe string[] GetLabels()
		{
			CheckModelLoaded();
			
			IntPtr labelsPtr;
			int numLabels = GetLabels(_fastText, new IntPtr(&labelsPtr));

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
		/// Predicts a single label from input text.
		/// </summary>
		/// <param name="text">Text to predict a label from.</param>
		/// <returns>Single prediction.</returns>
		public unsafe Prediction PredictSingle(string text)
		{
			CheckModelLoaded();
			CheckModelLabels();

			IntPtr labelPtr;
			float prob = PredictSingle(_fastText, _utf8.GetBytes(text), new IntPtr(&labelPtr));

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

			int cnt = PredictMultiple(_fastText, _utf8.GetBytes(text), new IntPtr(&labelsPtr), probs, number);
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
			int dim = GetSentenceVector(_fastText, _utf8.GetBytes(text), new IntPtr(&vecPtr));

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
		/// Trains a new supervised classification model.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
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

			TrainSupervised(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix);
			_maxLabelLen = GetMaxLabelLenght(_fastText);
			_modelLoaded = true;
		}

		/// <summary>
		/// Trains a new model using low-level FastText arguments.
		/// </summary>
		/// <param name="inputPath">Path to a training set.</param>
		/// <param name="outputPath">Path to write the model to (excluding extension).</param>
		/// <param name="args">Low-level training arguments.</param>
		/// <remarks>Trained model will consist of two files: .bin (main model) and .vec (word vectors).</remarks>
		public void Train(string inputPath, string outputPath, FastTextArgs args)
		{
			ValidatePaths(inputPath, outputPath, args.PretrainedVectors);

			var argsStruct = new TrainingArgsStruct
							{
								bucket = args.bucket,
								cutoff = args.cutoff,
								dim = args.dim,
								dsub = args.dsub,
								epoch = args.epoch,

								loss = (loss_name)args.loss,
								lr = args.lr,
								lrUpdateRate = args.lrUpdateRate,
								maxn = args.maxn,
								minCount = args.minCount,
								minCountLabel = args.minCountLabel,
								minn = args.minn,
								model = (model_name)args.model,
								neg = args.neg,

								qnorm = args.qnorm,
								qout = args.qout,
								retrain = args.retrain,
								saveOutput = args.saveOutput,
								t = args.t,
								thread = args.thread,
								verbose = args.verbose,
								wordNgrams = args.wordNgrams,
								ws = args.ws,
							};

			Train(_fastText, inputPath, outputPath, argsStruct, args.LabelPrefix, args.PretrainedVectors);
			_maxLabelLen = GetMaxLabelLenght(_fastText);
			_modelLoaded = true;
		}

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

		private void CheckModelLabels()
		{
			if (_maxLabelLen == 0)
			{
				throw new InvalidOperationException("Loaded model doesn't contain supervised labels. Maybe you loaded an unsupervised model?");
			}
		}
		
		private void CheckModelLoaded()
		{
			if (!_modelLoaded)
			{
				throw new InvalidOperationException("Model not loaded!");
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
