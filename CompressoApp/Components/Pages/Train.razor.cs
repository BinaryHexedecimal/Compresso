using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using System.Text.Json;
using Microsoft.JSInterop;
using System.Diagnostics;
//using Microsoft.Extensions.Configuration;
//using Microsoft.Extensions.DependencyInjection;


namespace CompressoApp.Components.Pages;


public partial class Train : ComponentBase, IDisposable
{

    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private TrainStateService TrainState { get; set; } = default!;
    //[Inject] private SummaryLoadService SummaryService { get; set; } = default!;
    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;
    //[Inject] private IConfiguration Configuration { get; set; } = default!;
    //[Inject] private BackendUrls BackendUrls { get; set; } = default!;  // ✅ Direct injection

    [Inject] private IServiceProvider Services { get; set; } = default!;
    private string backendUrl = string.Empty;
    private BackendUrls? backendUrls;

    private DotNetObjectReference<Train>? dotnetRef;
    private object? sseInstance;

    private CancellationTokenSource? cts;
    private Timer? elapsedTimer;
    private Stopwatch? trainStopwatch;

    private List<CompressionSummary> summariesFromContainer { get; set; } = new List<CompressionSummary>();
    private List<CompressionSummary> allSummaries { get; set; } = new List<CompressionSummary>();
    private bool firstRowFromDefault = false;

    private int selectedEpoch = -1;
    private string SaveMessage = "";
    private bool HasSavedModel = false;

    //private List<SavedModelInfo> SavedModels = new();

    //private string Mode = "NothingYet";  //"SavedModelMode" or "SetParameterMode"

    //private string? SelectedSavedModelId { get; set; } = "";


    protected override async Task OnInitializedAsync()
    {

        //var backendUrls = Services.GetRequiredService<BackendUrls>();

        backendUrls = Services.GetRequiredService<BackendUrls>();

        backendUrl = backendUrls.External;
        // Create a timer that checks every 1 second
        elapsedTimer = new Timer(async _ =>
        {
            // backendUrl =
            // Environment.GetEnvironmentVariable("BACKEND_URL")
            // ?? Configuration["Backend:BaseUrl"]
            // ?? "http://127.0.0.1:8000";

            //var backendUrls = Services.GetRequiredService<BackendUrls>();



            if (TrainState.IsTraining || TrainState.IsTerminating)
            {
                if (trainStopwatch == null || !trainStopwatch.IsRunning)
                    trainStopwatch = Stopwatch.StartNew();

                TrainState.ElapsedSeconds = (int)trainStopwatch.Elapsed.TotalSeconds;
                await InvokeAsync(StateHasChanged);
            }
            else
            {
                trainStopwatch?.Stop();
                trainStopwatch = null;
            }
        }, null, 0, 1000); // run immediately, tick every 1000ms

        dotnetRef = DotNetObjectReference.Create(this);
        if (NavManager.Uri.EndsWith("/new"))
        {
            ClearTrainState();
            ClearSetting();
            ClearSummaries();
            //ClearModels();
        }
        summariesFromContainer = await Api.LoadAllSummariesFromContainerAsync();
        if (TrainState.DefaultSummary != null)
        {
            allSummaries.Add(TrainState.DefaultSummary);
            firstRowFromDefault = true;
        }
        allSummaries.AddRange(summariesFromContainer);



        //SavedModels = await Api.GetSavedModelInfoAsync(); // Example service call

    }


    private void SelectData(string selectedId)
    {
        TrainState.FinalDataId = selectedId;
        TrainState.FinalSummary = allSummaries?.FirstOrDefault(d => d.CompressionJobId == selectedId);
        StateHasChanged();
    }

    private void SelectEpoch(int epoch)
    {
        selectedEpoch = epoch;
        StateHasChanged();
    }

    // private async Task SaveModel()
    // {
    //     if (selectedEpoch == -1)
    //     {
    //         SaveMessage = "⚠️Please select an epoch to save.";
    //         await ShowSaveMessageAsync(SaveMessage);
    //         return;
    //     }

    //     var resultMessage = await Api.SaveModelAsync(selectedEpoch, TrainState.CurrentTrainId);

    //     SaveMessage = resultMessage;     
    //     HasSavedModel = true;

    //     await ShowSaveMessageAsync(SaveMessage);
    //     await InvokeAsync(StateHasChanged);
    // }


    private async Task SaveModel()
    {
        if (selectedEpoch == -1)
        {
            SaveMessage = "⚠️Please select an epoch to save.";
            await ShowSaveMessageAsync(SaveMessage);
            return;
        }

        var info = new SavedModelInfo
        {
            ModelId = Guid.NewGuid().ToString(),
            DatasetName = TrainState.FinalSummary?.DatasetName ?? "unknown",
            K = TrainState.FinalSummary?.K ?? -1,
            Kind = TrainState.SelectedTrainingType ?? "standard"
        };

        var resultMessage = await Api.SaveModelAsync(selectedEpoch, TrainState.CurrentTrainId, info);

        SaveMessage = resultMessage;
        HasSavedModel = true;
        //update available models

        //SavedModels = await Api.GetSavedModelInfoAsync();

        await ShowSaveMessageAsync(SaveMessage);

        await InvokeAsync(StateHasChanged);
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


    private async Task StartTrain()
    {
        TrainState.IsPreparingForTraining = true;
        TrainState.IsTraining = true;

        cts = new CancellationTokenSource();

        // Generate a new UUID / GUID
        Guid newGuid = Guid.NewGuid();
        string trainJobId = newGuid.ToString();
        TrainState.CurrentTrainId = trainJobId;
        BaseTrainRequest? req = null;
        if (TrainState.SelectedTrainingType == "Standard" && TrainState.FinalSummary != null)
        {

            req = new StandardTrainRequest
            {
                TrainJobId = trainJobId,
                Kind = "standard",
                DataInfo = new Dictionary<string, string>{
                                            { "dataset_name", TrainState.FinalSummary.DatasetName },
                                            { "k", TrainState.FinalSummary.K.ToString() },
                                            { "norm", TrainState.FinalSummary.Norm },
                                            { "data_id", TrainState.FinalSummary.CompressionJobId },
                },
                DataId = TrainState.FinalDataId,
                Optimizer = TrainState.StandardOptimizer,
                NumIterations = TrainState.StandardItr,
                LearningRate = TrainState.StandardLr,
                RequireAdvAttackTest = TrainState.RequireFinalAdvAttackTest,
            };

        }
        else if (TrainState.SelectedTrainingType == "Adversarial" && TrainState.FinalSummary != null)
        {
            req = new AdvTrainRequest
            {
                TrainJobId = trainJobId,
                Kind = "adversarial",
                DataInfo = new Dictionary<string, string>{
                                            { "dataset_name", TrainState.FinalSummary.DatasetName },
                                            { "k", TrainState.FinalSummary.K.ToString() },
                                            { "norm", TrainState.FinalSummary.Norm },
                                            { "data_id", TrainState.FinalSummary.CompressionJobId },
                },
                DataId = TrainState.FinalDataId,
                Optimizer = TrainState.AdvOptimizer,
                NumIterations = TrainState.AdvItr,
                LearningRate = TrainState.AdvLr,
                Attack = TrainState.AdvAttack,
                Epsilon = TrainState.AdvEps,
                Alpha = TrainState.AdvAlpha,
                RequireAdvAttackTest = TrainState.RequireFinalAdvAttackTest,
            };

        }

        var reqJson = JsonSerializer.Serialize(req);
        dotnetRef ??= DotNetObjectReference.Create(this);

        // Call JS to start SSE POST
        sseInstance = await JS.InvokeAsync<object>(
            "startSSEPost",
            //"http://127.0.0.1:8000/train",
            //"http://backend:8000",
            //$"{backendUrl}/train",
            $"{backendUrls!.External}/train",
            reqJson,
            dotnetRef
        );

    }


    public void Dispose()
    {
        dotnetRef?.Dispose();
    }

    [JSInvokable]
    public void ReceiveSSEMessage(object msg)
    {
        if (msg == null) return;

        try
        {
            if (msg is not JsonElement je) return;

            if (!je.TryGetProperty("type", out var typeElem)) return;
            string type = typeElem.GetString() ?? "";
            switch (type)
            {
                case "start":
                    TrainState.IsPreparingForTraining = false;
                    TrainState.IsTraining = true;
                    break;

                case "epoch":
                    // Build metrics dictionary
                    var metrics = new Dictionary<string, object>();

                    foreach (var prop in je.EnumerateObject())
                    {
                        if (prop.Name == "type") continue;
                        metrics[prop.Name] = prop.Value.ValueKind switch
                        {
                            JsonValueKind.Number when prop.Value.TryGetDouble(out double d) => d,
                            JsonValueKind.String => prop.Value.GetString() ?? "",
                            JsonValueKind.True => true,
                            JsonValueKind.False => false,
                            _ => prop.Value.ToString() ?? ""
                        };
                    }
                    TrainState.EpochMetrics.Add(metrics);
                    break;

                case "done":
                    TrainState.IsTraining = false;
                    TrainState.HasCompleted = true;
                    break;

                case "error":
                    ClearTrainState();
                    ClearSetting();
                    string errMsg = je.TryGetProperty("error", out var err)
                        ? err.GetString() ?? "unknown"
                        : "unknown";
                    Console.WriteLine("Training error: " + errMsg);
                    break;

                case "cancelled":
                    TrainState.IsTerminating = false;
                    TrainState.HasTerminated = true;
                    break;

                default:
                    Console.WriteLine("Unknown SSE type: " + type);
                    break;
            }
            InvokeAsync(StateHasChanged);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to process SSE message: " + ex);
        }
    }


    private async Task CancelTraining()
    {
        if (!TrainState.IsTraining || cts == null || string.IsNullOrEmpty(TrainState.CurrentTrainId)) return;
        cts.Cancel();
        await Api.CancelTrainingAsync(TrainState.CurrentTrainId);
        TrainState.IsTraining = false;
        TrainState.IsTerminating = true;
        StateHasChanged();
    }

    private string FormatValue(object value)
    {
        if (value is double d)
        {
            return d.ToString("F3"); // 3 decimal places
        }
        return value?.ToString() ?? "";
    }

    private void GoBackToCompression()
    {
        NavManager.NavigateTo($"/Compression");
    }




    private void ClearTrainState()
    {
        TrainState.IsTraining = false;
        TrainState.IsPreparingForTraining = false;
        TrainState.HasCompleted = false;

        TrainState.IsTerminating = false;
        TrainState.HasTerminated = false;

        TrainState.EpochMetrics.Clear();
        TrainState.ElapsedSeconds = 0;

        TrainState.FinalDataId = "";
        TrainState.FinalSummary = null;
        TrainState.CurrentTrainId = "";


        TrainState.SelectedTrainingType = "";
        TrainState.RequireFinalAdvAttackTest = false;

        TrainState.StandardOptimizer = "SGD";
        TrainState.StandardItr = 10;
        TrainState.StandardLr = 0.01;

        // Adversarial Training
        TrainState.AdvAttack = "PGD-linf";
        TrainState.AdvEps = 0.3;
        TrainState.AdvOptimizer = "Adam";
        TrainState.AdvItr = 10;
        TrainState.AdvLr = 0.01;
        TrainState.AdvAlpha = 0.01;

        StateHasChanged();
    }

    private void ClearSummaries()
    {
        summariesFromContainer?.Clear();
        allSummaries.Clear();
        firstRowFromDefault = false;
        StateHasChanged();
    }

    // private void ClearModels()
    // {
    //     //SavedModels.Clear();
    //     //IsUsingSavedModel = false;
    //     StateHasChanged();

    // }
    private void ClearSetting()
    {
        selectedEpoch = -1;
        SaveMessage = "";
        HasSavedModel = false;
        //SelectedSavedModelId = string.Empty;
        //Mode = "NothingYet";
        StateHasChanged();
    }





    // private void OnSavedModelChanged(ChangeEventArgs e)
    // {
    //     SelectedSavedModelId = e.Value?.ToString() ?? "";

    //     if (!string.IsNullOrEmpty(SelectedSavedModelId))
    //         Mode = "SavedModelMode";
    //     else if (!string.IsNullOrEmpty(TrainState.SelectedTrainingType))
    //         Mode = "SetParameterMode";
    //     else
    //         Mode = "NothingYet";
    // }


    // private void OnTrainingTypeChanged(ChangeEventArgs e)
    // {
    //     TrainState.SelectedTrainingType = e.Value?.ToString() ?? "";

    //     if (!string.IsNullOrEmpty(TrainState.SelectedTrainingType))
    //         Mode = "SetParameterMode";
    //     else if (!string.IsNullOrEmpty(SelectedSavedModelId))
    //         Mode = "SavedModelMode";
    //     else
    //         Mode = "NothingYet";
    // }



    private async Task NewTrainingAsync()
    {
        // Clean local state


        // Ask backend to delete checkpoints
        if (!string.IsNullOrEmpty(TrainState.CurrentTrainId))
        {
            var msg = await Api.DeleteCheckpointsAsync(TrainState.CurrentTrainId);
            Console.WriteLine(msg);
        }
        ClearTrainState();
        ClearSetting();

        //await InvokeAsync(StateHasChanged);
    }


}    






