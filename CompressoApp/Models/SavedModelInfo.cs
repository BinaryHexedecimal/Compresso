using System.Text.Json.Serialization;
using System.Text.Json;

namespace CompressoApp.Models;

public class SavedModelInfo
{
    [JsonPropertyName("model_id")]
    public string ModelId { get; set; } = "";

    [JsonPropertyName("dataset_name")]
    public string DatasetName { get; set; } = "";
    [JsonPropertyName("k")]
    public int K { get; set; } = -1;
    [JsonPropertyName("kind")]
    public string Kind { get; set; } = "standard";

}
