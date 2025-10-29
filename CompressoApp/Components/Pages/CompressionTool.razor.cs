using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Base64 = System.String;
//using Label = System.String;
using System.Text.Json;
using System.Diagnostics;

namespace CompressoApp.Components.Pages;

public partial class CompressionTool : ComponentBase, IDisposable
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;
    [Inject] private ImageService ImageService { get; set; } = default!;
    //[Inject] private CompressionStateService CompressionState { get; set; } = default!;
    //[Inject] private TrainStateService TrainState { get; set; } = default!;
    [Inject] private DatasetInfoService DatasetInfoManager { get; set; } = default!;
    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private IServiceProvider Services { get; set; } = default!;
    private string backendUrl = string.Empty;
    private BackendUrls? backendUrls;



    public string? CompressionId { get; set; }

    private const int numImagesPerRow = 8;
    private Stopwatch? compressionStopwatch;
    //private CancellationTokenSource? cts;
    private Timer? elapsedTimer;
    private string KError = "";
    private string EtaError = "";


    // show center
    private bool showCenterImages = false;
    private Dictionary<string, List<Base64>> centerImages = new();


    // save state
    private string? SaveMessage { get; set; }
    private bool ShowSaveOptions { get; set; } = false;
    private bool HasSaved { get; set; } = false;
    private string CandidateDuplicateId { get; set; } = "";




    private DotNetObjectReference<CompressionTool>? dotnetRef;
    private object? sseInstance;




    //-----copy from  state
    //public string CompressionId { get; set; } = string.Empty;

    // compression settings
    public string DatasetName { get; set; } = "";
    public string Norm { get; set; } = "L2";
    public int K { get; set; } = 10;
    public double? Eta { get; set; } = null;
    public string Optimizer { get; set; } = "gurobi";


    // compression progress
    public int ElapsedSeconds { get; set; } = 0;
    public int Progress { get; set; } = 0;
    public int Total { get; set; } = 10;
    public bool IsCompressing { get; set; } = false;
    public bool IsPreparingForCompression { get; set; } = false;
    public bool HasFinished { get; set; } = false;
    public bool IsCancelling { get; set; } = false;
    //public bool HasCancelled { get; set; } = false;
    public string ProgressPercent => Total > 0 ? $"{Progress * 100 / Total}%" : "0%";

    // images
    public Dictionary<string, List<string>> Images { get; set; } = new();

    // compression summary
    public CompressionSummary? CompressionSummary { get; set; }









    protected async override void OnInitialized()
    {
        // if (NavManager.Uri.EndsWith("/new"))
        // {
        Clear();
        var msg = await Api.DeleteAllGraphDataAsync();
        Console.WriteLine(msg);
        // }

        backendUrls = Services.GetRequiredService<BackendUrls>();
        backendUrl = backendUrls.External;


        dotnetRef = DotNetObjectReference.Create(this);


        // Create a timer that checks every 2 second
        elapsedTimer = new Timer(async _ =>
        {
            if (IsCompressing)
            {
                if (compressionStopwatch == null || !compressionStopwatch.IsRunning)
                    compressionStopwatch = Stopwatch.StartNew();

                ElapsedSeconds = (int)compressionStopwatch.Elapsed.TotalSeconds;
                await InvokeAsync(StateHasChanged);
            }
            else
            {
                compressionStopwatch?.Stop();
                compressionStopwatch = null;
            }
        }, null, 0, 2000); // run immediately, tick every 2000ms

        await DatasetInfoManager.RefreshFromBackendAsync();
    }



    private void StateClear()
    {
        Progress = 0;
        Total = 10;
        IsPreparingForCompression = false;
        ElapsedSeconds = 0;
        CompressionId = "";
        IsCompressing = false;
        Images.Clear();
        HasFinished = false;
        //HasCancelled = false;
        CompressionSummary = null;
        IsCancelling = false;

        DatasetName = "";
        Norm = "L2";
        K = 10;
        Eta = null;
        Optimizer = "gurobi";

        StateHasChanged();
    }
    private void SettingClear()
    {
        SaveMessage = "";
        CandidateDuplicateId = "";
        KError = "";
        EtaError = "";
        showCenterImages = false;
        centerImages.Clear();
        ShowSaveOptions = false;
        HasSaved = false;

        StateHasChanged();
    }


    private void Clear()
    {
        StateClear();
        SettingClear();
    }

    void OnDatasetChanged(string newDatasetName)
    {
        if (string.IsNullOrWhiteSpace(newDatasetName))
        {
            DatasetName = string.Empty;
            Total = 0;
            return;
        }
        Clear();

        DatasetName = newDatasetName;
        Total = DatasetInfoManager.Labels[DatasetName].Count;
    }


    private async Task StartCompression()
    {
        IsCompressing = true;
        IsPreparingForCompression = true;
        //cts = new CancellationTokenSource();
        Guid newGuid = Guid.NewGuid();
        CompressionId = newGuid.ToString();
        var req = new CompressRequest
        {
            CompressionJobId = CompressionId,
            OriginDatasetName = DatasetName,
            K = K,
            Eta = Eta ?? 0.0,
            Norm = Norm,
            Optimizer = Optimizer
        };





        var reqJson = JsonSerializer.Serialize(req);

        sseInstance = await JS.InvokeAsync<object>(
            "startSSEPost",
            $"{backendUrls!.External}/compress",
            reqJson,
            dotnetRef
        );

    }


    [JSInvokable]
    public async Task ReceiveSSEMessage(JsonElement message)
    {

        if (message.TryGetProperty("progress", out var prog))
        {
            Progress = prog.GetInt32();
            if (Progress == 0)
            {
                IsPreparingForCompression = false;
                IsCompressing = true;
            }
            StateHasChanged();
        }
        if (message.TryGetProperty("total", out var _total))
        {
            Total = _total.GetInt32();
            StateHasChanged();
        }

        if (message.TryGetProperty("type", out var typeEl))
        {
            var type = typeEl.GetString();
            if (type == "start")
            {
                IsPreparingForCompression = true;
                
                StateHasChanged();
            }
            if (type == "done")
            {
                HasFinished = true;
                IsCompressing = false;
                Progress = Total;
                //HasFinished = true;
                StateHasChanged();
                await Task.Delay(500);
                await InvokeAsync(StateHasChanged);
                try
                {
                    CompressionSummary = await Api.GetCompressionSummaryFromMemoryAsync(CompressionId!);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error fetching summary: {ex}");
                }
                await InvokeAsync(StateHasChanged);
            }
            else if (type == "cancelled")
            {
                // handle cancellation
                //if (!HasFinished)
                Console.WriteLine("hold da op!");
                if (!string.IsNullOrEmpty(CompressionId))
                    {
                        var msg = await Api.DeleteGraphDataAsync(CompressionId);
                        Console.WriteLine(msg);
                    }
                Clear();
                StateHasChanged();

            }
        }
    }





    private async Task CancelCompression()
    {
        if (!IsCompressing || string.IsNullOrEmpty(CompressionId)) 
            return;
        IsCancelling = true;
        IsCompressing = false;
        await Api.CancelCompressionAsync(CompressionId);
        //await Task.Delay(200); 
    }

















    private void Nav(string path) => NavManager.NavigateTo(path);

    public void Dispose()
    {
        elapsedTimer?.Dispose();
    }


    private void ValidateK(ChangeEventArgs e)
    {
        if (int.TryParse(e.Value?.ToString(), out int value))
        {
            if (value < 1 || value > 500)
                KError = "k must between 1 and 500.";
            else
                KError = "";
        }
        else
        {
            KError = "k must be an integer.";
        }
    }

    private void ValidateEta(ChangeEventArgs e)
    {
        if (double.TryParse(e.Value?.ToString(), out double value))
        {
            if (value < 0)
                EtaError = "eta must be positive.";
            else
                EtaError = "";
        }
        else
        {
            EtaError = "eta must be a number.";
        }
    }


    private async Task ToggleCenterFigures()
    {
        showCenterImages = !showCenterImages;
        if (showCenterImages
            && CompressionSummary != null
            && centerImages.Count == 0)
        {
            string jobId = CompressionSummary.CompressionJobId;
            var labels = DatasetInfoManager.Labels[DatasetName];
            bool origin = false;
            centerImages = await ImageService.FetchImages(jobId, labels, numImagesPerRow, origin);
            StateHasChanged();
        }

    }
    private async Task RefreshBatch(string label)
    {
        if (showCenterImages && CompressionSummary != null)
        {
            bool origin = false;
            string jobId = CompressionSummary.CompressionJobId;
            centerImages[label] = await ImageService.FetchImagesForOneLabel(jobId, label, numImagesPerRow, origin);
            StateHasChanged();
        }
    }

    private async Task SaveToContainer()
    {
        if (string.IsNullOrEmpty(CompressionId))
        {
            SaveMessage = "⚠️No compression ID available.";
            await ShowSaveMessageAsync(SaveMessage, false);
            return;
        }

        var response = await Api.SaveCompressionAsync(CompressionId);
        SaveMessage = response.SaveMessage;
        if (response.RequireUserDecision && response!.SaveMessage!.Contains("Duplicate"))
        {
            ShowSaveOptions = true;
            CandidateDuplicateId = response.DuplicateId!;
            await InvokeAsync(StateHasChanged);
            await ShowSaveMessageAsync(response.SaveMessage, false);
        }
        else if (response.RequireUserDecision && response!.SaveMessage!.Contains("full"))
        {
            ShowSaveOptions = true;
            await InvokeAsync(StateHasChanged);
            await ShowSaveMessageAsync(response.SaveMessage, false);
        }
        else if (response.SaveMessage == "✅Saved successfully.")
        {
            HasSaved = true;
            await InvokeAsync(StateHasChanged);
            await ShowSaveMessageAsync(response.SaveMessage);
        }
        else
        {
            await ShowSaveMessageAsync("⚠️Unknown response from server.", false);
        }
        return;
    }


    private async Task HandleReplaceChoice()
    {
        if (string.IsNullOrEmpty(CompressionId))
        {
            SaveMessage = "⚠️No compression ID available.";
            await ShowSaveMessageAsync(SaveMessage, false);
            return;
        }
        var result = await Api.HandleReplaceAsync(CompressionId, CandidateDuplicateId);

        CandidateDuplicateId = "";
        SaveMessage = result.SaveMessage;
        ShowSaveOptions = false;
        HasSaved = true;
        await ShowSaveMessageAsync(SaveMessage!);
    }


    public async Task HandleCancelChoice()
    {
        SaveMessage = "⚠️Save canceled";
        ShowSaveOptions = false;
        await ShowSaveMessageAsync(SaveMessage);
    }

    private void HandleManageChoice()
    {
        ShowSaveOptions = false;
        Nav($"/Container");
    }

    // private void GoToTrain()
    // {
    //     Nav($"/Train");
    //     DefaultDataId = CompressionId!;
    //     DefaultSummary = CompressionSummary;

    // }
    private void ShowGraph(string label)
    {
        var url = $"/graph/{CompressionId}/{DatasetName}/{label}/{K}";
        JS.InvokeVoidAsync("window.open", url, "_blank");
    }


    private async Task ShowSaveMessageAsync(string message, bool autoClear = true, int delayMs = 6000)
    {
        SaveMessage = message;
        StateHasChanged();

        if (autoClear)
        {
            await Task.Delay(delayMs);
            SaveMessage = string.Empty;
            await InvokeAsync(StateHasChanged);
        }
    }

    private async Task NewCompressionAsync()
    {
        // Clean local state

        Console.WriteLine("called Newcpression async ???");
        // Ask backend to delete checkpoints
        if (!string.IsNullOrEmpty(CompressionId))
        {
            var msg = await Api.DeleteGraphDataAsync(CompressionId);
            Console.WriteLine(msg);
        }
        Console.WriteLine(CompressionId);
        Clear();

        //await InvokeAsync(StateHasChanged);
    }

}






     







