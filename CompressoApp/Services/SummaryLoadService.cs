// using CompressoApp.Models;
// using Microsoft.AspNetCore.Components;

// namespace CompressoApp.Services;

// public class SummaryLoadService
// {
//     private readonly ApiClient _api;
//     public SummaryLoadService(ApiClient api)
//     {
//         _api = api;
//     }

//     public async Task<List<CompressionSummary>> LoadAllSummariesFromContainerAsync()
//     {
//         try
//         {
//             var summaries = await _api.GetAllSummariesFromContainerAsync();
//             return summaries;
//         }
//         catch
//         {
//             return new List<CompressionSummary>();
//         }
//     }
//     public async Task<CompressionSummary> LoadSummaryFromContainerAsync(string dataId)
//     {
//         var summary = await _api.GetSummaryFromContainerAsync(dataId);
//         return summary;
//     }
// }
