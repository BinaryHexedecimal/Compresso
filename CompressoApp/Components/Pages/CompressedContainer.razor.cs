using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;


namespace CompressoApp.Components.Pages;

public partial class CompressedContainer: ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;
    //[Inject] private SummaryLoadService SummaryService { get; set; } = default!;
    [Inject] private TrainStateService TrainState { get; set; } = default!;

    private List<CompressionSummary>? summaries;

    protected override async Task OnInitializedAsync()
    {
        summaries = await Api.LoadAllSummariesFromContainerAsync();
    }


    // private async Task<bool> LoadSummariesAsync()
    // {
    //     // Load all summaries from container
    //     var latest = await SummaryService.LoadAllSummariesFromContainerAsync();

    //     // Sort descending by Timestamp
    //     latest = latest.OrderByDescending(s => s.Timestamp).ToList();

    //     // Compare with existing list
    //     if (summaries == null || !latest.SequenceEqual(summaries, new SummaryComparer()))
    //     {
    //         summaries = latest;
    //         return true;
    //     }

    //     return false;
    // }


    // public class SummaryComparer : IEqualityComparer<CompressionSummary>
    // {
    //     public bool Equals(CompressionSummary? x, CompressionSummary? y)
    //     {
    //         if (x == null && y == null) return true;
    //         if (x == null || y == null) return false;

    //         // Compare by JobId (or Timestamp, or both)
    //         return x.CompressionJobId == y.CompressionJobId;
    //     }

    //     public int GetHashCode(CompressionSummary obj)
    //     {
    //         return HashCode.Combine(obj.CompressionJobId);
    //     }
    // }


    private async Task HandleTrain(string jobId)
    {
        // Optional: Show a status message or refresh summaries
        Nav($"/Train/new");
        TrainState.DefaultDataId = jobId;
        TrainState.DefaultSummary = await Api.LoadSummaryFromContainerAsync(jobId);

    }

    private void Nav(string path) => NavManager.NavigateTo(path);

    private async Task HandleDelete(string jobId)
    {
        var resultMessage = await Api.DeleteContainerDataAsync(jobId);
        // Refresh container after deletion
        summaries = await Api.LoadAllSummariesFromContainerAsync();

        await InvokeAsync(StateHasChanged);
    }

    private async Task ConfirmDeleteAll()
    {
        bool confirm = await JS.InvokeAsync<bool>(
            "confirm",
            new object[] { "⚠️ This will delete ALL data permanently. Are you sure?" }
        );
        if (!confirm)
            return;
        var result = await Api.DeleteAllContainerDataAsync();
        // Refresh after deletion

        //modelInfos = await Api.GetSavedModelInfoAsync();
        summaries = await Api.LoadAllSummariesFromContainerAsync();
        await InvokeAsync(StateHasChanged);

        Console.WriteLine(result);
    }

}