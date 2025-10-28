// using CompressoApp.Models; 
// public class CompressionStateService
// {
//     public string CompressionId { get; set; } = string.Empty;

//     // compression settings
//     public string DatasetName { get; set; } = "";
//     public string Norm { get; set; } = "L2";
//     public int K { get; set; } = 10;
//     public double? Eta { get; set; } = null;
//     public string Optimizer { get; set; } = "gurobi";


//     // compression progress
//     public int ElapsedSeconds { get; set; } = 0;
//     public int Progress { get; set; } = 0;
//     public int Total { get; set; } = 10;
//     public bool IsCompressing { get; set; } = false;
//     public bool IsPreparingForCompression { get; set; } = false;
//     public bool HasFinished { get; set; } = false;
//     public bool IsCancelling { get; set; } = false;
//     //public bool HasCancelled { get; set; } = false;
//     public string ProgressPercent => Total > 0 ? $"{Progress * 100 / Total}%" : "0%";

//     // images
//     public Dictionary<string, List<string>> Images { get; set; } = new();

//     // compression summary
//     public CompressionSummary? CompressionSummary { get; set; }

// }
