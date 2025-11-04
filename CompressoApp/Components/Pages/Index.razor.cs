using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Label = System.String;

namespace CompressoApp.Components.Pages;

public partial class Index : ComponentBase
{
    [Inject] private ImageService ImageService { get; set; } = default!;
    [Inject] private DatasetInfoService DatasetInfoManager { get; set; } = default!;
    
    private int numImagesPerRow = 14;
    private bool showImages = false;
    private Dictionary<string, List<string>> Images { get; set; } = new();
    private string datasetName = string.Empty;
    private string datasetDescription { get; set; } = string.Empty;

    private List<string> defaultDatasets = new() { "mnist", "cifar10", "cifar100", "svhn" };

    private void OnDatasetChanged(string newDatasetName)
    {
        datasetName = newDatasetName;

        if (defaultDatasets.Contains(newDatasetName))
        {
            datasetDescription = DatasetInfoManager.Descriptions[datasetName];
        }
        else
        {
            datasetDescription = "User-uploaded dataset";
        }

        showImages = false;
        Images.Clear();
        StateHasChanged();
    }

    private async Task ToggleFigures()
    {
        showImages = !showImages;

        if (showImages && Images.Count == 0)
        {
            if (!DatasetInfoManager.Labels.ContainsKey(datasetName))
            {
                Console.WriteLine($"No labels found for dataset '{datasetName}'");
                return;
            }

            var labels = DatasetInfoManager.Labels[datasetName];

            if (labels == null || labels.Count == 0)
            {
                Console.WriteLine($"Empty label list for dataset '{datasetName}'");
                return;
            }

            Images = await ImageService.FetchImages(datasetName, labels, numImagesPerRow, true);
        }

        StateHasChanged();
    }

    private async Task RefreshBatch(Label label)
    {
        if (!string.IsNullOrEmpty(datasetName))
        {
            Images[label] = await ImageService.FetchImagesForOneLabel(datasetName, label, numImagesPerRow, true);
            StateHasChanged();
        }
    }


    protected async override void OnInitialized()
    {
        showImages = false;
        Images = new();
        datasetName = string.Empty;
        datasetDescription = string.Empty;
        await DatasetInfoManager.RefreshFromBackendAsync();
    }
}
