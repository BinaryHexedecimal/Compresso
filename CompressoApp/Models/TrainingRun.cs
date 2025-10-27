using System.Text.Json.Serialization;
using CompressoApp.Services;

namespace CompressoApp.Models;

public class TrainingRun
{
    [JsonPropertyName("train_job_id")]
    public string TrainJobId { get; set; } = "";
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }
    [JsonPropertyName("status")]
    public string Status { get; set; } = ""; // "done" | "cancelled" | "error"
    [JsonPropertyName("req_obj")]
    public required BaseTrainRequest TrainRequest { get; set; }

    [JsonPropertyName("epochs")]
    public List<EpochMetrics> Epochs { get; set; } = new();

}


public class EpochMetrics
{
    [JsonPropertyName("epoch")]
    public int Epoch { get; set; }

    [JsonPropertyName("train_acc")]
    public double TrainAcc { get; set; }

    [JsonPropertyName("test_acc")]
    public double TestAcc { get; set; }

    [JsonPropertyName("linf_adv_acc")]
    public double LinfAdvAcc { get; set; }

    [JsonPropertyName("l2_adv_acc")]
    public double L2AdvAcc { get; set; }
}
