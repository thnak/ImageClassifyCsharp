using System.Diagnostics;
using System.Text.Json;
using Microsoft.JSInterop;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Caching.Memory;


namespace ImageClassify
{
    public enum ModelWeight
    {
        MobileNet_V3_Small_Weights,
        MobileNet_V3_Large_Weights,
        Resnet18
    }

    public enum DeviceTypes
    {
        Android,
        IOS,
        MacOs,
        MacCatalist,
        WinUi,
        Tizen
    }

    public interface IImageClassifyService
    {
        Task<Dictionary<string, float>> InferenceAsync(MemoryStream memoryStream);
        Task<Dictionary<string, float>> InferenceAsync(Image<Rgb24> image);
        Task<Dictionary<string, float>> InferenceAsync(Stream stream);
        Task<Dictionary<string, float>> InferenceAsync(string imagePath);
        Task<Dictionary<string, float>> InferenceAsync(byte[] byteImage);
        void Dispose();
    }

    public class ImageClassifyService : IImageClassifyService, IDisposable
    {
        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly List<string> _categories;
        private readonly IEnumerable<string> _inputNames;
        private readonly IEnumerable<string> _outputNames;
        private bool _disposed = false;
        private IMemoryCache _MemoryCache { get; }

        private readonly int[] _inputShape;
        private IJSRuntime? _jsRuntime;

        public ImageClassifyService(IMemoryCache memoryCache)
        {
            _MemoryCache = memoryCache;
            if (_MemoryCache.TryGetValue("_sessionOptions", out SessionOptions sessionOptions))
            {
                if (sessionOptions is not null) _sessionOptions = sessionOptions;
            }

            if (_MemoryCache.TryGetValue("_session", out InferenceSession inferenceSession))
            {
                if (inferenceSession is not null) _session = inferenceSession;
            }

            if (_MemoryCache.TryGetValue("_runOptions", out RunOptions runOptions))
            {
                if (runOptions is not null) _runOptions = runOptions;
            }

            if (_MemoryCache.TryGetValue("_jsRuntime", out IJSRuntime jsRuntime))
            {
                _jsRuntime = jsRuntime;
            }
            JsLogger("[ImageClassifyService][_MemoryCache] reuse form memory");
        }

        public ImageClassifyService(ModelWeight weight, DeviceTypes device, IJSRuntime? jsRuntime = null)
        {
            if (jsRuntime != null)
            {
                _jsRuntime = jsRuntime;
            }

            _sessionOptions = new SessionOptions();
            _sessionOptions.EnableMemoryPattern = true;
            _sessionOptions.EnableProfiling = false;
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            ProviderOptionsValueHelper helper = new ProviderOptionsValueHelper();

            switch (device)
            {
                case DeviceTypes.Android:
                {
                    try
                    {
                        NnapiFlags nnapiFlags = NnapiFlags.NNAPI_FLAG_USE_NCHW;
                        _sessionOptions.AppendExecutionProvider_Nnapi(nnapiFlags);
                        JsLogger($"[ImageClassifyService][Init][NNAPI_FLAG_USE_NCHW]");
                    }
                    catch (Exception e)
                    {
                        JsLogger($"[ImageClassifyService][Init] {e.Message}");
                    }

                    break;
                }
                case DeviceTypes.IOS:
                {
                    try
                    {
                        _sessionOptions.AppendExecutionProvider_CoreML(CoreMLFlags
                            .COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
                        JsLogger($"[ImageClassifyService][Init][AppendExecutionProvider_CoreML]");
                    }
                    catch (Exception e)
                    {
                        JsLogger($"[ImageClassifyService][Init] {e.Message}");
                    }

                    break;
                }
                case DeviceTypes.MacOs:
                {
                    try
                    {
                        _sessionOptions.AppendExecutionProvider_CoreML(CoreMLFlags
                            .COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
                    }
                    catch (Exception e)
                    {
                        JsLogger($"[ImageClassifyService][Init][AppendExecutionProvider_CoreML]");
                    }

                    break;
                }
                case DeviceTypes.WinUi:
                {
                    try
                    {
                        _sessionOptions.EnableMemoryPattern = false;
                        _sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                        _sessionOptions.AppendExecutionProvider_DML(0);
                        JsLogger($"[ImageClassifyService][Init][AppendExecutionProvider_DML]");
                    }
                    catch (Exception e)
                    {
                        JsLogger($"[ImageClassifyService][Init] {e.Message}");
                    }

                    break;
                }
                default:
                {
                    break;
                }
            }

            var prepackedWeightsContainer = new PrePackedWeightsContainer();
            _runOptions = new RunOptions();

            switch (weight)
            {
                case ModelWeight.MobileNet_V3_Large_Weights:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_large, _sessionOptions,
                        prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_large");
                    break;
                }
                case ModelWeight.MobileNet_V3_Small_Weights:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_small, _sessionOptions,
                        prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_small");

                    break;
                }
                case ModelWeight.Resnet18:
                {
                    _session = new InferenceSession(Properties.Resources.resnet18, _sessionOptions,
                        prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] resnet18");

                    break;
                }
                default:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_small, _sessionOptions,
                        prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_small");

                    break;
                }
            }

            var metadata = _session.ModelMetadata;
            var customMetadata = metadata.CustomMetadataMap;
            string categories;
            if (customMetadata.TryGetValue("categories", out categories))
            {
                _categories = new List<string>(JsonSerializer.Deserialize<List<string>>(categories));
            }

            _inputNames = _session.InputNames;
            _outputNames = _session.OutputNames;
            _inputShape = _session.InputMetadata.First().Value.Dimensions;
            _MemoryCache = new MemoryCache(new MemoryCacheOptions());
            _MemoryCache.Set("_runOptions", _runOptions);
            _MemoryCache.Set("_sessionOptions", _sessionOptions);
            _MemoryCache.Set("_session", _session);
            if (_jsRuntime is not null) _MemoryCache.Set("_jsRuntime", _jsRuntime);
        }

        public async Task<Dictionary<string, float>> InferenceAsync(MemoryStream memoryStream)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(memoryStream);
            var result = await InferenceAsync(image);
            return result;
        }

        public async Task<Dictionary<string, float>> InferenceAsync(Stream stream)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(stream);
            return await InferenceAsync(image);
        }

        public async Task<Dictionary<string, float>> InferenceAsync(string imagePath)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
            return await InferenceAsync(image);
        }

        public async Task<Dictionary<string, float>> InferenceAsync(byte[] byteImage)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(byteImage);
            return await InferenceAsync(image);
        }

        public async Task<Dictionary<string, float>> InferenceAsync(Image<Rgb24> image)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(_inputShape[2], _inputShape[3]),
                    Mode = ResizeMode.Crop
                });
            });
            
            DenseTensor<float> processedImage = new(new[] { 1, 3, 224, 224 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        processedImage[0, 0, y, x] = (float)pixelSpan[x].R;
                        processedImage[0, 1, y, x] = (float)pixelSpan[x].G;
                        processedImage[0, 2, y, x] = (float)pixelSpan[x].B;
                    }
                }
            });

            long[] sLongs = _inputShape.Select(item => (long)item).ToArray();
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, processedImage.Buffer, sLongs);
            var inputs = new Dictionary<string, OrtValue>
            {
                { _inputNames.First(), inputOrtValue }
            };
            using IDisposableReadOnlyCollection<OrtValue> predictions = await Task.FromResult(_session.Run(_runOptions, inputs, _session.OutputNames));
            Dictionary<string, float> resultDic = new Dictionary<string, float>();
            var scores = predictions.First().Value.GetTensorDataAsSpan<float>().ToArray();
            var indies = predictions.Last().Value.GetTensorDataAsSpan<long>().ToArray();
            for (int i = 0; i < indies.Length; i++)
            {
                resultDic.Add(_categories[(int)indies[i]], scores[i]);
            }

            stopwatch.Stop();
            JsLogger($"[ImageClassifyService][InferenceAsync][SPEED] {stopwatch.ElapsedMilliseconds}ms");
            return resultDic;
        }

        private void JsLogger(string message)
        {
            if (_jsRuntime is not null)
            {
                _jsRuntime.InvokeVoidAsync("console.log", message);
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                _MemoryCache.Dispose();
                _sessionOptions.Dispose();
                _session.Dispose();
                _runOptions.Dispose();
                JsLogger("[ImageClassifyService][Dispose] disposed");
            }

            _disposed = true;
        }
    }
}