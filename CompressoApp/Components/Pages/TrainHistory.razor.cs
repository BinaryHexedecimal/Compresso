using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;


namespace CompressoApp.Components.Pages;

public partial class TrainHistory : ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    private List<TrainingRun>? history;
    private TrainingRun? selectedTrainingRun;
    private string highlightTrainJobId = string.Empty;

    protected override async Task OnInitializedAsync()
    {
        selectedTrainingRun = null;
        highlightTrainJobId = "";
        history = await Api.GetHistoryAsync();
    }

    private async Task HandleDelete(string trainId)
    {
        var resultMessage = await Api.DeleteTrainingRunAsync(trainId);
        history = await Api.GetHistoryAsync();
        await InvokeAsync(StateHasChanged);
    }

    private void ShowEpochs(string trainId)
    {
        selectedTrainingRun = history?.FirstOrDefault(h => h.TrainJobId == trainId);
        highlightTrainJobId = trainId;
        StateHasChanged();
        return;
    }

    private string FormatValue(object value)
    {
        if (value is double d)
        {
            return d.ToString("F3"); // 3 decimal places
        }
        return value?.ToString() ?? "";
    }


    private async Task ConfirmDeleteAll()
    {
        bool confirm = await JS.InvokeAsync<bool>(
            "confirm",
            new object[] { "This will delete ALL history permanently. Are you sure?" }
        );

        if (!confirm)
            return;

        var result = await Api.DeleteAllHistoryAsync();

        // Refresh after deletion
        history = await Api.GetHistoryAsync();
        await InvokeAsync(StateHasChanged);

        Console.WriteLine(result);
    }

}