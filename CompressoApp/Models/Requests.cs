using System.Text.Json.Serialization;
using System.Text.Json;

namespace CompressoApp.Services;


public class CompressRequest
{
    [JsonPropertyName("compression_job_id")]
    public string CompressionJobId { get; set; } = "";

    [JsonPropertyName("dataset_name")]
    public string OriginDatasetName { get; set; } = "";

    [JsonPropertyName("k")]
    public int K { get; set; }

    [JsonPropertyName("eta")]
    public double Eta { get; set; }

    [JsonPropertyName("norm")]
    public string Norm { get; set; } = "";

    [JsonPropertyName("optimizer")]
    public string Optimizer { get; set; } = "";

}

public class EvaluationRequest
{
    [JsonPropertyName("evaluation_id")]
    public string EvaluationId { get; set; } = default!;
    [JsonPropertyName("dataset_name")]
    public string DatasetName { get; set; } = default!;
    [JsonPropertyName("model_id")]
    public string ModelId { get; set; } = default!;
    [JsonPropertyName("train_")]
    public bool Train_ { get; set; } = false;  // defaults to test
}



public abstract class BaseTrainRequest
{
    [JsonPropertyName("kind")]
    public string Kind { get; set; } = "";
    [JsonPropertyName("train_job_id")]
    public string TrainJobId { get; set; } = "";

    [JsonPropertyName("data_info")]
    public Dictionary<string, string> DataInfo { get; set; } = new Dictionary<string, string> { }; 

    [JsonPropertyName("data_id")]
    public string DataId { get; set; } = "";

    [JsonPropertyName("optimizer")]
    public string Optimizer { get; set; } = "SGD";

    [JsonPropertyName("num_iterations")]
    public int NumIterations { get; set; } = 5;

    [JsonPropertyName("learning_rate")]
    public double LearningRate { get; set; } = 0.01;
    [JsonPropertyName("require_adv_attack_test")]
    public bool RequireAdvAttackTest { get; set; } = false;
}

public class StandardTrainRequest : BaseTrainRequest { }

public class AdvTrainRequest : BaseTrainRequest
{
    [JsonPropertyName("attack")]
    public string Attack { get; set; } = "PGD-linf";

    [JsonPropertyName("epsilon")]
    public double Epsilon { get; set; } = 0.3;

    [JsonPropertyName("alpha")]
    public double Alpha { get; set; } = 0.01;
}



public class BaseTrainRequestConverter : JsonConverter<BaseTrainRequest>
{
    public override BaseTrainRequest Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        if (!doc.RootElement.TryGetProperty("kind", out var kindProp))
            throw new JsonException("Missing 'kind' property");

        var kind = kindProp.GetString();

        return kind switch
        {
            "standard" => JsonSerializer.Deserialize<StandardTrainRequest>(doc.RootElement.GetRawText(), options)!,
            "adversarial" => JsonSerializer.Deserialize<AdvTrainRequest>(doc.RootElement.GetRawText(), options)!,
            _ => throw new JsonException($"Unknown kind: {kind}")
        };
    }

    public override void Write(Utf8JsonWriter writer, BaseTrainRequest value, JsonSerializerOptions options)
    {
        JsonSerializer.Serialize(writer, (object)value, options);
    }
}