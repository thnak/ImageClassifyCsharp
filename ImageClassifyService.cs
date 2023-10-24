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
        Tizen,
        Default
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
            _sessionOptions.EnableCpuMemArena = true;
            _sessionOptions.EnableProfiling = false;
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;

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
                    _sessionOptions.EnableMemoryPattern = true;
                    _sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                    JsLogger($"[ImageClassifyService][Init][DEFAULT]");
                    break;
                }
            }

            var prepackedWeightsContainer = new PrePackedWeightsContainer();
            _runOptions = new RunOptions();

            switch (weight)
            {
                case ModelWeight.MobileNet_V3_Large_Weights:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_large, _sessionOptions, prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_large");
                    break;
                }
                case ModelWeight.MobileNet_V3_Small_Weights:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_small, _sessionOptions, prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_small");
                    break;
                }
                case ModelWeight.Resnet18:
                {
                    _session = new InferenceSession(Properties.Resources.resnet18, _sessionOptions, prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] resnet18");
                    break;
                }
                default:
                {
                    _session = new InferenceSession(Properties.Resources.mobilenet_v3_small, _sessionOptions, prepackedWeightsContainer);
                    JsLogger($"[ImageClassifyService][Init] mobilenet_v3_small");
                    break;
                }
            }

            var metadata = _session.ModelMetadata;
            var customMetadata = metadata.CustomMetadataMap;
            if (customMetadata.TryGetValue("categories", out var categories))
            {
                var content = JsonSerializer.Deserialize<List<string>>(categories);
                if (content != null) _categories = new List<string>(content);
                else
                {
                    JsLogger("[ImageClassifyService][Init][ERROR] not found categories in model metadata");
                    _categories = new List<string>();
                    for (var i = 0; i < 10000; i++)
                    {
                        _categories.Add($"Named[{i}]");
                    }
                }
            }
            else
            {
                JsLogger("[ImageClassifyService][Init][ERROR] not found categories in model metadata");
                _categories = new List<string>();
                for (var i = 0; i < 10000; i++)
                {
                    _categories.Add($"Named[{i}]");
                }
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
            using Image<Rgb24> image = Image.Load<Rgb24>(byteImage);
            return await InferenceAsync(image);
        }

        public async Task<Dictionary<string, float>> InferenceAsync(Image<Rgb24> image)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(_inputShape[2], _inputShape[3]),
                    Mode = ResizeMode.Crop
                });
            });

            var processedImage = new DenseTensor<float>(dimensions: _inputShape);
            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < accessor.Height; y++)
                {
                    var pixelSpan = accessor.GetRowSpan(y);
                    for (var x = 0; x < accessor.Width; x++)
                    {
                        processedImage[0, 0, y, x] = pixelSpan[x].R;
                        processedImage[0, 1, y, x] = pixelSpan[x].G;
                        processedImage[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });

            var sLongs = _inputShape.Select(item => (long)item).ToArray();
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, processedImage.Buffer, sLongs);
            var inputs = new Dictionary<string, OrtValue> { { _inputNames.First(), inputOrtValue } };

            using IDisposableReadOnlyCollection<OrtValue> predictions = await Task.FromResult(_session.Run(_runOptions, inputs, _session.OutputNames));

            var resultDic = new Dictionary<string, float>();
            var scores = predictions[0].Value.GetTensorDataAsSpan<float>().ToArray();
            var indies = predictions[1].Value.GetTensorDataAsSpan<long>().ToArray();
            for (var i = 0; i < indies.Length; i++)
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