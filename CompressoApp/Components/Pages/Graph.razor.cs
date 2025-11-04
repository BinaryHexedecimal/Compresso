using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;


namespace CompressoApp.Components.Pages;

public partial class Graph : ComponentBase, IDisposable
{

    [Inject] private IJSRuntime JS { get; set; } = default!;
    [Inject] private ApiClient Api { get; set; } = default!;
    [Parameter] public string jobId { get; set; } = default!;
    [Parameter] public string label { get; set; } = default!;
    [Parameter] public int k { get; set; } = 0;
    [Parameter] public string datasetName { get; set; } = ""!;

    private ElementReference _graphDivRef;
    private string _pendingFigJson = string.Empty;
    private DotNetObjectReference<Graph>? dotNetRef;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (!firstRender) { return; }
        dotNetRef = DotNetObjectReference.Create(this);
        await JS.InvokeVoidAsync("setDotNetRefForGraph", dotNetRef);
        
        // Get raw JSON string from the API client
        var figJson = await Api.GetGraphJsonAsync(jobId, label, k);

        if (!string.IsNullOrWhiteSpace(figJson))
        {
            _pendingFigJson = figJson;
            await Task.Delay(200); 
            await JS.InvokeVoidAsync("ensurePlotlyReadyAndRender", _graphDivRef, _pendingFigJson, label);
            StateHasChanged();
        }

    }



    public void Dispose()
    {
        dotNetRef?.Dispose();
        JS.InvokeVoidAsync("clearDotNetRefForGraph");
        Console.WriteLine($"{GetType().Name} disposed.");
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
