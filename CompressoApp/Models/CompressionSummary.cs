using System.Text.Json.Serialization;

namespace CompressoApp.Models;


public class CompressionSummary
{
    [JsonPropertyName("compression_id")]
    public string CompressionJobId { get; set; } = string.Empty;

    [JsonPropertyName("dataset_name")]
    public string DatasetName { get; set; } = string.Empty;
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }
    [JsonPropertyName("norm")]
    public string Norm { get; set; } = string.Empty;

    [JsonPropertyName("k")]
    public int K { get; set; }

    [JsonPropertyName("elapsed_seconds")]
    public int ElapsedSeconds { get; set; }
    [JsonPropertyName("labels")]
    public List<string> Labels { get; set; } = [];
}
