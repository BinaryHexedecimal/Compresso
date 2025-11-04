using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using System.Text.Json;
using Microsoft.JSInterop;
using System.Diagnostics;

namespace CompressoApp.Components.Pages;

public partial class Train : ComponentBase, IDisposable
{

    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private IServiceProvider Services { get; set; } = default!;
    private string backendUrl = string.Empty;
    private BackendUrls? backendUrls;
    private DotNetObjectReference<Train>? dotnetRef;

    private Stopwatch? trainStopwatch;
    private Timer? elapsedTimer;
    private List<CompressionSummary> summaries { get; set; } = new List<CompressionSummary>();


    private int selectedEpoch = -1;
    private string SaveMessage = "";
    private bool HasSavedModel = false;
    //private bool IsSaving = false;



    //state

    private string CurrentTrainId { get; set; } = "";
    private string FinalDataId { get; set; } = "";
    private CompressionSummary? FinalSummary { get; set; }


    // Train settings
    private string SelectedTrainingType { get; set; } = "";
    private bool RequireFinalAdvAttackTest { get; set; } = false;


    // Standard Training
    private string StandardOptimizer { get; set; } = "SGD";
    private int StandardItr { get; set; } = 10;
    private double StandardLr { get; set; } = 0.01;

    // Adversarial Training
    private string AdvAttack { get; set; } = "PGD-linf";
    private double AdvEps { get; set; } = 0.3;
    private string AdvOptimizer { get; set; } = "Adam";
    private int AdvItr { get; set; } = 10;
    private double AdvLr { get; set; } = 0.01;
    private double AdvAlpha { get; set; } = 0.01;



    // training progress
    private int ElapsedSeconds { get; set; } = 0;
    private bool IsTraining { get; set; } = false;
    private bool HasCompleted { get; set; } = false;
    private bool IsPreparingForTraining { get; set; } = false;
    private bool IsTerminating { get; set; } = false;
    private bool HasTerminated { get; set; } = false;

    // result
    private record struct EpochMetrics(int Epoch, double TrainAcc, double TestAcc, double AdvAcc);
    private List<EpochMetrics> EpochMetricsList { get; set; } = new List<EpochMetrics>();



    protected override async Task OnInitializedAsync()
    {

        backendUrls = Services.GetRequiredService<BackendUrls>();
        backendUrl = backendUrls.External;

        // Create a timer that checks every 1 second
        elapsedTimer = new Timer(async _ =>
        {
            if (IsTraining || IsTerminating)
            {
                if (trainStopwatch == null || !trainStopwatch.IsRunning)
                    trainStopwatch = Stopwatch.StartNew();

                ElapsedSeconds = (int)trainStopwatch.Elapsed.TotalSeconds;
                await InvokeAsync(StateHasChanged);
            }
            else
            {
                trainStopwatch?.Stop();
                trainStopwatch = null;
            }
        }, null, 0, 1000); // run immediately, tick every 1000ms

        dotnetRef = DotNetObjectReference.Create(this);

        ClearTrainState();
        ClearSetting();
        ClearSummaries();

        summaries = await Api.LoadAllSummariesFromContainerAsync();

        StateHasChanged();

    }


    private void SelectData(string selectedId)
    {
        FinalDataId = selectedId;
        FinalSummary = summaries?.FirstOrDefault(d => d.CompressionJobId == selectedId);
        StateHasChanged();
    }

    private void SelectEpoch(int epoch)
    {
        selectedEpoch = epoch;
        StateHasChanged();
    }



    private async Task SaveModel()
    {
        if (selectedEpoch == -1)
        {
            SaveMessage = "Please select an epoch to save.";
            await ShowSaveMessageAsync(SaveMessage);
            return;
        }

        var info = new SavedModelInfo
        {
            ModelId = Guid.NewGuid().ToString(),
            DatasetName = FinalSummary?.DatasetName ?? "unknown",
            K = FinalSummary?.K ?? -1,
            Kind = SelectedTrainingType ?? "standard",
            TestAcc = EpochMetricsList[selectedEpoch].TestAcc
        };
        //IsSaving = true;
        StateHasChanged();
        var resultMessage = await Api.SaveModelAsync(selectedEpoch+1, CurrentTrainId, info);

        SaveMessage = resultMessage;
        HasSavedModel = true;
        //IsSaving = false;
        StateHasChanged();
        await ShowSaveMessageAsync(SaveMessage);
        //await InvokeAsync(StateHasChanged);
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
        IsPreparingForTraining = true;
        IsTraining = true;

        // Generate a new UUID / GUID
        Guid newGuid = Guid.NewGuid();
        string trainJobId = newGuid.ToString();
        CurrentTrainId = trainJobId;
        BaseTrainRequest? req = null;
        if (SelectedTrainingType == "Standard" && FinalSummary != null)
        {
            req = new StandardTrainRequest
            {
                TrainJobId = trainJobId,
                Kind = "standard",
                DataInfo = new Dictionary<string, string>{
                                            { "dataset_name", FinalSummary.DatasetName },
                                            { "k", FinalSummary.K.ToString() },
                                            { "norm", FinalSummary.Norm },
                                            { "data_id", FinalSummary.CompressionJobId },
                },
                DataId = FinalDataId,
                Optimizer = StandardOptimizer,
                NumIterations = StandardItr,
                LearningRate = StandardLr,
                RequireAdvAttackTest = RequireFinalAdvAttackTest,
            };

        }
        else if (SelectedTrainingType == "Adversarial" && FinalSummary != null)
        {
            req = new AdvTrainRequest
            {
                TrainJobId = trainJobId,
                Kind = "adversarial",
                DataInfo = new Dictionary<string, string>{
                                            { "dataset_name", FinalSummary.DatasetName },
                                            { "k", FinalSummary.K.ToString() },
                                            { "norm", FinalSummary.Norm },
                                            { "data_id", FinalSummary.CompressionJobId },
                },
                DataId = FinalDataId,
                Optimizer = AdvOptimizer,
                NumIterations = AdvItr,
                LearningRate = AdvLr,
                Attack = AdvAttack,
                Epsilon = AdvEps,
                Alpha = AdvAlpha,
                RequireAdvAttackTest = RequireFinalAdvAttackTest,
            };

        }

        var reqJson = JsonSerializer.Serialize(req);
        dotnetRef ??= DotNetObjectReference.Create(this);

        // Call JS to start SSE POST
        await JS.InvokeAsync<object>(
            "startSSEPost",
            $"{backendUrl}/train",
            reqJson,
            dotnetRef
        );

    }


    public void Dispose()
    {
        dotnetRef?.Dispose();
        elapsedTimer?.Dispose();

        // Trigger SSE cleanup in JavaScript if needed
        _ = JS.InvokeVoidAsync("stopSSE", $"{backendUrl}/train");
        Console.WriteLine($"{GetType().Name} disposed.");
    }

    [JSInvokable]
    public void ReceiveSSEMessage(JsonElement message)
    {
        if (message.TryGetProperty("type", out var typeElement))
        {
            var type = typeElement.GetString();
            switch (type)
            {
                case "epoch":
                    int epoch = message.GetProperty("epoch").GetInt32();
                    double trainAcc = message.GetProperty("train_acc").GetDouble();
                    double testAcc = message.GetProperty("test_acc").GetDouble();
                    double linfAdvAcc = message.GetProperty("linf_adv_acc").GetDouble();
                    var metrics = new EpochMetrics(epoch, trainAcc, testAcc, linfAdvAcc);
                    EpochMetricsList.Add(metrics);
                    break;

                case "start":
                    IsPreparingForTraining = false;
                    IsTraining = true;
                    break;

                case "done":
                    IsTraining = false;
                    HasCompleted = true;
                    break;

                case "error":
                    ClearTrainState();
                    ClearSetting();
                    break;

                case "cancelled":
                    IsTerminating = false;
                    HasTerminated = true;
                    break;

                default:
                    Console.WriteLine("Unknown SSE type: " + type);
                    break;
            }
            InvokeAsync(StateHasChanged);
        }
    }


    private async Task CancelTraining()
    {
        if (!IsTraining || string.IsNullOrEmpty(CurrentTrainId)) return;
        IsTraining = false;
        IsTerminating = true;
        StateHasChanged();

        await Api.CancelTrainingAsync(CurrentTrainId);
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


    private void ClearTrainState()
    {
        IsTraining = false;
        IsPreparingForTraining = false;
        HasCompleted = false;

        IsTerminating = false;
        HasTerminated = false;

        EpochMetricsList.Clear();
        ElapsedSeconds = 0;

        FinalDataId = "";
        FinalSummary = null;
        CurrentTrainId = "";


        SelectedTrainingType = "";
        RequireFinalAdvAttackTest = false;

        StandardOptimizer = "SGD";
        StandardItr = 10;
        StandardLr = 0.01;

        // Adversarial Training
        AdvAttack = "PGD-linf";
        AdvEps = 0.3;
        AdvOptimizer = "Adam";
        AdvItr = 10;
        AdvLr = 0.01;
        AdvAlpha = 0.01;

        StateHasChanged();
    }

    private void ClearSummaries()
    {
        summaries.Clear();
        StateHasChanged();
    }

    private void ClearSetting()
    {
        selectedEpoch = -1;
        SaveMessage = "";
        HasSavedModel = false;
        //IsSaving = false;
        StateHasChanged();
    }


    private async Task NewTrainingAsync()
    {
        // Ask backend to delete checkpoints
        if (!string.IsNullOrEmpty(CurrentTrainId))
        {
            var msg = await Api.DeleteCheckpointsAsync(CurrentTrainId);
            Console.WriteLine(msg);
        }
        ClearTrainState();
        ClearSetting();
    }


}    






