using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;


namespace CompressoApp.Components.Pages;

public partial class Model: ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;
    //[Inject] private TrainStateService TrainState { get; set; } = default!;
    //[Inject] private SummaryLoadService SummaryService { get; set; } = default!;


    private List<SavedModelInfo>? modelInfos = new List<SavedModelInfo>();


    private Dictionary<string, double> acc = new Dictionary<string, double>();

    private string? evaluatingModelId = null;   // Track which model is currently being evaluated






    protected override async Task OnInitializedAsync()
    {
        modelInfos = await Api.GetSavedModelInfoAsync();
    }



    private void Nav(string path) => NavManager.NavigateTo(path);



    private async Task HandleDelete(string modelId)
    {
        var resultMessage = await Api.DeleteModelAsync(modelId);
        // Refresh container after deletion
        modelInfos = await Api.GetSavedModelInfoAsync();

        await InvokeAsync(StateHasChanged);
    }


    private async Task Evaluate(string modelId)
    {
        // Mark as evaluating
        evaluatingModelId = modelId;
        await InvokeAsync(StateHasChanged);

        try
        {
            var modelInfo = modelInfos!
                .FirstOrDefault(m => string.Equals(m.ModelId, modelId, StringComparison.OrdinalIgnoreCase));

            if (modelInfo != null)
            {
                var req = new EvaluationRequest
                {
                    EvaluationId = Guid.NewGuid().ToString(),
                    DatasetName = modelInfo.DatasetName,
                    ModelId = modelId,
                    Train_ = false
                };

                var accuracy = await Api.EvaluateModelAsync(req);

                if (accuracy.HasValue)
                {
                    acc[modelId] = accuracy.Value;
                    Console.WriteLine($"✅ Accuracy: {accuracy.Value:P2}");
                }
                else
                {
                    Console.WriteLine("❌ Evaluation failed.");
                }
            }
        }
        finally
        {
            // Clear loading state
            evaluatingModelId = null;
            await InvokeAsync(StateHasChanged);
        }
    }


    private async Task ConfirmDeleteAll()
    {
        bool confirm = await JS.InvokeAsync<bool>(
            "confirm",
            new object[] { "⚠️ This will delete ALL saved models permanently. Are you sure?" }
        );
        if (!confirm)
            return;
        var result = await Api.DeleteAllModelsAsync();
        // Refresh after deletion
        modelInfos = await Api.GetSavedModelInfoAsync();
        await InvokeAsync(StateHasChanged);

        Console.WriteLine(result);
    }


}