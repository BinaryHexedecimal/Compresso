namespace CompressoApp.Services;

public class ImageService
{
    private readonly ApiClient _api;

    public ImageService(ApiClient apiClient)
    {
        _api = apiClient;
    }

    public async Task<Dictionary<string, List<string>>> FetchImages(
                                                    string id, List<string> labels,
                                                    int numImagesPerRow, bool origin)
    {
        var images = new Dictionary<string, List<string>>();
        var tasks = new List<Task>();
        foreach (var label in labels)
        {
            tasks.Add(Task.Run(async () =>
            {
                List<string> imgs;
                if (origin)
                    imgs = await _api.GetOriginImagesAsync(id, label, numImagesPerRow);
                else
                    imgs = await _api.GetCompressedImagesAsync(id, label, numImagesPerRow);

                lock (images)  // protect concurrent writes
                {
                    images[label] = imgs;
                }
            }));
        }
        await Task.WhenAll(tasks);
        return images;
    }

    public async Task<List<string>> FetchImagesForOneLabel(string id,
                                            string label, int numImagesPerRow, bool origin)
    {
        if (origin)
        {
            return await _api.GetOriginImagesAsync(id, label, numImagesPerRow);
        }
        else
        {
            return await _api.GetCompressedImagesAsync(id, label, numImagesPerRow);

        }
    }
    
}

