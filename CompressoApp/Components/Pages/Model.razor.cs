using CompressoApp.Models;
using CompressoApp.Services;
using Microsoft.JSInterop;
using Microsoft.AspNetCore.Components;


namespace CompressoApp.Components.Pages;

public partial class Model: ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private IServiceProvider Services { get; set; } = default!;

    private string backendUrl = string.Empty;
    private BackendUrls? backendUrls;

    private List<SavedModelInfo>? modelInfos = new List<SavedModelInfo>();


    protected override async Task OnInitializedAsync()
    {
        modelInfos = await Api.GetSavedModelInfoAsync();
        
        backendUrls = Services.GetRequiredService<BackendUrls>();
        backendUrl = backendUrls.External;
    }

    private async Task HandleDelete(string modelId)
    {
        var resultMessage = await Api.DeleteModelAsync(modelId);
        // Refresh container after deletion
        modelInfos = await Api.GetSavedModelInfoAsync();

        await InvokeAsync(StateHasChanged);
    }


    private async Task DownloadModel(string modelId)
    {
        var mi = modelInfos!.First(m => string.Equals(m.ModelId, modelId, StringComparison.OrdinalIgnoreCase));
        var desired = $"trained_model_{mi.DatasetName}_{mi.Kind}_K_{mi.K}.pt";

        var apiUrl = $"{backendUrl}/download_model/{modelId}?display_name={Uri.EscapeDataString(desired)}";
        await JS.InvokeVoidAsync("downloadFileFromUrl", apiUrl, desired);
    }


    private async Task ConfirmDeleteAll()
    {
        bool confirm = await JS.InvokeAsync<bool>(
            "confirm",
            new object[] { "This will delete ALL saved models permanently. Are you sure?" }
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