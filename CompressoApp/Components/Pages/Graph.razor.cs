using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using System.Text.Json;

namespace CompressoApp.Components.Pages;

public partial class Graph : ComponentBase
{

    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private NavigationManager NavManager { get; set; } = default!;

    [Parameter] public string jobId { get; set; } = default!;
    [Parameter] public string label { get; set; } = default!;
    [Parameter] public int k { get; set; } = 0;
    [Parameter] public string datasetName { get; set; } = ""!;

    private ElementReference _graphDivRef;
    private string _pendingFigJson= string.Empty;

    private void Nav(string path) => NavManager.NavigateTo(path);

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (!firstRender) { return; }
        var _dotNetRef = DotNetObjectReference.Create(this);
        await JS.InvokeVoidAsync("setDotNetRefForGraph", _dotNetRef);

        var response = await Api.GetGraphJsonAsync(jobId,label,k);
        var json = await response.Content.ReadFromJsonAsync<JsonElement>();

        if (json.TryGetProperty("fig_json", out var figProp))
        {
            _pendingFigJson = figProp.GetString()?? string.Empty;
            await Task.Delay(300); 
            await JS.InvokeVoidAsync("ensurePlotlyReadyAndRender", _graphDivRef, _pendingFigJson, label);
        }

    }



    [JSInvokable]
    public async Task OnCenterNodeClicked(int nodeIndex, double x, double y, string label)
    {
        Console.WriteLine($"Center node clicked: {nodeIndex}");
        var imageUrl = await Api.GetNodeImageAsync(jobId, label, nodeIndex);
        // Clear previous center image (if any)
        await JS.InvokeVoidAsync("clearNodeImage", _graphDivRef, "centerImage");
        // Render new image
        await JS.InvokeVoidAsync("addNodeImage", _graphDivRef, imageUrl, x, y, "centerImage");
    }

    [JSInvokable]
    public async Task OnNodeHovered(int nodeIndex, double x, double y, string label)
    {
        Console.WriteLine($"Hovered node: {nodeIndex}");
        var imageUrl = await Api.GetNodeImageAsync(jobId, label, nodeIndex);
        await JS.InvokeVoidAsync("addNodeImage", _graphDivRef, imageUrl, x, y, "hoverImage");
    }
}
