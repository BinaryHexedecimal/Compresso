using System.Text.Json.Serialization;
namespace CompressoApp.Models;

public class ProgressUpdate
{
    [JsonPropertyName("progress")]
    public int? Progress { get; set; }

    [JsonPropertyName("done")]
    public bool? Done { get; set; }
    [JsonPropertyName("label")]
    public string? Label { get; set; }

    [JsonPropertyName("total")]
    public int? Total { get; set; }
    [JsonPropertyName("start")]
    public bool? Start { get; set; }

}
