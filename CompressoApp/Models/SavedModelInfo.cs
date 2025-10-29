using System.Text.Json.Serialization;


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
    [JsonPropertyName("test_acc")]
    public double TestAcc { get; set; } = -1;

}
