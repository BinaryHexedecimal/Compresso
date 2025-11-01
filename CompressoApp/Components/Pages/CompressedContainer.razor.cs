using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;

namespace CompressoApp.Components.Pages;

public partial class CompressedContainer : ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;
    [Inject] private IServiceProvider Services { get; set; } = default!;
    private string backendUrl = string.Empty;
    private BackendUrls? backendUrls;

    private List<CompressionSummary>? summaries;

    protected override async Task OnInitializedAsync()
    {
        summaries = await Api.LoadAllSummariesFromContainerAsync();
        
        backendUrls = Services.GetRequiredService<BackendUrls>();
        backendUrl = backendUrls.External;
    }



    private void HandleTrain()
    {
        Nav($"/Train");
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
            new object[] { "This will delete ALL data permanently. Are you sure?" }
        );
        if (!confirm)
            return;
        var result = await Api.DeleteAllContainerDataAsync();

        summaries = await Api.LoadAllSummariesFromContainerAsync();
        await InvokeAsync(StateHasChanged);

        //Console.WriteLine(result);
    }


    private async Task HandleDownload(string compressionJobId)
    {
        var sm = summaries!.First(s => string.Equals(s.CompressionJobId, compressionJobId, StringComparison.OrdinalIgnoreCase));
        var desired = $"compressed_data_{sm.DatasetName}_{sm.Norm}_K_{sm.K}.pt";

        var apiUrl = $"{backendUrls!.External}/download_compressed_data/{compressionJobId}?display_name={Uri.EscapeDataString(desired)}";

        await JS.InvokeVoidAsync("downloadFileFromUrl", apiUrl, desired);
    }
}