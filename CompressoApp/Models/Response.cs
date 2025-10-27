using System.Text.Json.Serialization;
namespace CompressoApp.Models;

public class CompressResponse
{
    public string JobId { get; set; } = string.Empty;
}

public class SaveResponse
{
    public string? SaveMessage { get; set; }
    public bool RequireUserDecision { get; set; }
    public string? DuplicateId { get; set; }
}


public class SummaryResponse
{
    public CompressionSummary? Summary { get; set; }
    public string Status { get; set; } = string.Empty;
}

public class StartCompressionResponse
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
}