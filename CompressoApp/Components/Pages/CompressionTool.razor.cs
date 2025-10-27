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
    [Inject] private CompressionStateService CompressionState { get; set; } = default!;
    [Inject] private TrainStateService TrainState { get; set; } = default!;
    [Inject] private DatasetInfoService DatasetInfoManager { get; set; } = default!;

    [Parameter]
    public string? CompressionId { get; set; }

    private const int numImagesPerRow = 8;
    private Stopwatch? compressionStopwatch;
    private CancellationTokenSource? cts;
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


    protected async override void OnInitialized()
    {
        if (NavManager.Uri.EndsWith("/new"))
        {
            Clear();
            var msg = await Api.DeleteAllGraphDataAsync();
            Console.WriteLine(msg);
        }

        // Create a timer that checks every 2 second
        elapsedTimer = new Timer(async _ =>
        {
            if (CompressionState.IsCompressing)
            {
                if (compressionStopwatch == null || !compressionStopwatch.IsRunning)
                    compressionStopwatch = Stopwatch.StartNew();

                CompressionState.ElapsedSeconds = (int)compressionStopwatch.Elapsed.TotalSeconds;
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
        CompressionState.Progress = 0;
        CompressionState.Total = 10;
        CompressionState.IsPreparingForCompression = false;
        CompressionState.ElapsedSeconds = 0;
        CompressionState.CompressionId = "";
        CompressionState.IsCompressing = false;
        CompressionState.Images.Clear();
        CompressionState.HasFinished = false;
        //CompressionState.HasCancelled = false;
        CompressionState.CompressionSummary = null;

        CompressionState.DatasetName = "";
        CompressionState.Norm = "L2";
        CompressionState.K = 10;
        CompressionState.Eta = null;
        CompressionState.Optimizer = "gurobi";

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
            CompressionState.DatasetName = string.Empty;
            CompressionState.Total = 0;
            return;
        }
        Clear();

        CompressionState.DatasetName = newDatasetName;
        CompressionState.Total = DatasetInfoManager.Labels[CompressionState.DatasetName].Count;
    }


    private async Task StartCompression()
    {
        CompressionState.IsCompressing = true;
        CompressionState.IsPreparingForCompression = true;
        cts = new CancellationTokenSource();
        Guid newGuid = Guid.NewGuid();
        CompressionState.CompressionId = newGuid.ToString();
        var req = new CompressRequest
        {
            CompressionJobId = CompressionState.CompressionId,
            OriginDatasetName = CompressionState.DatasetName,
            K = CompressionState.K,
            Eta = CompressionState.Eta ?? 0.0,
            Norm = CompressionState.Norm,
            Optimizer = CompressionState.Optimizer
        };

        var startResponse = await Api.StartCompressionAsync(req);
        // Start streaming progress
        //await StreamProgressAsync(CompressionState.CompressionId, cts.Token);
        // Fire-and-forget streaming; don't await
        if (startResponse?.Success == true)
        {
            await Task.Delay(1000);
            _ = Task.Run(async () =>
            {
                try
                {
                    await StreamProgressAsync(CompressionState.CompressionId, cts.Token);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Stream error: {ex}");
                }
            });
        }
        else
        {
            Console.WriteLine("Failed to start compression.");
            CompressionState.IsCompressing = false;
        }

    }

    private async Task StreamProgressAsync(string compressionId, CancellationToken token)
    {
        try
        {
            using var stream = await Api.GetStreamCompressionAsync(compressionId, token);
            using var reader = new StreamReader(stream);
            while (!reader.EndOfStream && !token.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync(token);
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data:"))
                {
                    await Task.Yield(); // <-- yield even if line ignored
                    continue;
                }
                var json = line.Substring(5).Trim();
                var update = JsonSerializer.Deserialize<ProgressUpdate>(json);
                if (update == null) continue;
                else if (update.Start == true)
                {
                    CompressionState.IsPreparingForCompression = false;
                    await InvokeAsync(StateHasChanged);
                }
                else if (update?.Progress != null)
                {
                    CompressionState.Progress = update.Progress.Value;
                    await InvokeAsync(StateHasChanged);
                    await Task.Yield();
                }
                else if (update?.Done == true)
                {
                    await Task.Delay(500);
                    CompressionState.IsCompressing = false;
                    CompressionState.Progress = CompressionState.Total;
                    CompressionState.HasFinished = true;
                    await InvokeAsync(StateHasChanged);
                    try
                    {
                        CompressionState.CompressionSummary = await Api.GetCompressionSummaryFromMemoryAsync(CompressionState.CompressionId);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error fetching summary: {ex}");
                    }
                    await InvokeAsync(StateHasChanged);
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            Clear();
            await InvokeAsync(StateHasChanged);
        }
    }

    private async Task CancelCompression()
    {
        if (!CompressionState.IsCompressing
                                || cts == null
                                || string.IsNullOrEmpty(CompressionState.CompressionId)
                                ) return;
        cts.Cancel();
        await Api.CancelCompressionAsync(CompressionState.CompressionId);
        await Task.Delay(2000); 
        if (!CompressionState.HasFinished)
        {   
            if (!string.IsNullOrEmpty(CompressionState.CompressionId))
                {
                    var msg = await Api.DeleteGraphDataAsync(CompressionState.CompressionId);
                    Console.WriteLine(msg);
                }
            Clear();
            StateHasChanged();
        }
                


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
            && CompressionState.CompressionSummary != null
            && centerImages.Count == 0)
        {
            string jobId = CompressionState.CompressionSummary.CompressionJobId;
            var labels = DatasetInfoManager.Labels[CompressionState.DatasetName];
            bool origin = false;
            centerImages = await ImageService.FetchImages(jobId, labels, numImagesPerRow, origin);
            StateHasChanged();
        }

    }
    private async Task RefreshBatch(string label)
    {
        if (showCenterImages && CompressionState.CompressionSummary != null)
        {
            bool origin = false;
            string jobId = CompressionState.CompressionSummary.CompressionJobId;
            centerImages[label] = await ImageService.FetchImagesForOneLabel(jobId, label, numImagesPerRow, origin);
            StateHasChanged();
        }
    }

    private async Task SaveToContainer()
    {
        if (string.IsNullOrEmpty(CompressionState.CompressionId))
        {
            SaveMessage = "⚠️No compression ID available.";
            await ShowSaveMessageAsync(SaveMessage, false);
            return;
        }

        var response = await Api.SaveCompressionAsync(CompressionState.CompressionId);
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
        if (string.IsNullOrEmpty(CompressionState.CompressionId))
        {
            SaveMessage = "⚠️No compression ID available.";
            await ShowSaveMessageAsync(SaveMessage, false);
            return;
        }
        var result = await Api.HandleReplaceAsync(CompressionState.CompressionId, CandidateDuplicateId);

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

    private void GoToTrain()
    {
        Nav($"/Train");
        TrainState.DefaultDataId = CompressionState.CompressionId!;
        TrainState.DefaultSummary = CompressionState.CompressionSummary;

    }
    private void ShowGraph(string label)
    {
        var url = $"/graph/{CompressionState.CompressionId}/{CompressionState.DatasetName}/{label}/{CompressionState.K}";
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
        if (!string.IsNullOrEmpty(CompressionState.CompressionId))
        {
            var msg = await Api.DeleteGraphDataAsync(CompressionState.CompressionId);
            Console.WriteLine(msg);
        }
        Console.WriteLine(CompressionState.CompressionId);
        Clear();

        //await InvokeAsync(StateHasChanged);
    }

}






     







