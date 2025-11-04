using CompressoApp.Models;
using System.Net.Http.Headers;
using System.Text.Json;

namespace CompressoApp.Services;

public class ApiClient
{
    private readonly HttpClient _http;

    public ApiClient(HttpClient http)
    {
        _http = http;
    }


    // --------------------- Compression -------------------------//

    public async Task<bool> GetGurobiStatusAsync()
    {
        try
        {
            using var response = await _http.GetAsync("/gurobi-status");
            if (!response.IsSuccessStatusCode)
                return false;

            var content = await response.Content.ReadAsStringAsync();

            using var doc = JsonDocument.Parse(content);
            if (doc.RootElement.TryGetProperty("gurobi_valid", out var value))
            {
                return value.GetBoolean();
            }

            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error checking Gurobi status: {ex.Message}");
            return false;
        }
    }

    public async Task CancelCompressionAsync(string compression_job_id)
    {
        Console.WriteLine("here send msg to backend to cancel");
        await _http.DeleteAsync($"/cancel_compression/{compression_job_id}");
    }


    public async Task<int> RequireOriginalDasasetMinSizePerLabelAsync(string dataset_name)
    {
        var url = $"/fetch_origin_dataset_min_size_per_label/{dataset_name}";
        var result = await _http.GetFromJsonAsync<int?>(url);
        Console.WriteLine(result);
        return result ?? -1;
    }


    // --------------------- Image support -------------------------//
    public async Task<List<string>> GetOriginImagesAsync(string id, string label, int n)
    {
        var url = $"/sample_origin_images?dataset_name={id}&label={label}&n={n}";
        var result = await _http.GetFromJsonAsync<List<string>>(url);
        return result ?? new List<string>();
    }

    public async Task<List<string>> GetCompressedImagesAsync(string compression_job_id, string label, int n)
    {
        var url = $"/sample_compressed_images?compression_job_id={compression_job_id}&label={label}&n={n}";
        var result = await _http.GetFromJsonAsync<List<string>>(url);
        return result ?? new List<string>();
    }

    public async Task<string> GetClusterImageAsync(string compression_job_id, string label)
    {
        var url = $"plot_cluster/{compression_job_id}/{label}";
        using var response = await _http.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var bytes = await response.Content.ReadAsByteArrayAsync();
        // Convert to base64 string for <img src="">
        string base64 = Convert.ToBase64String(bytes);
        return $"data:image/png;base64,{base64}";
    }

    public async Task<string> GetNodeImageAsync(string compression_job_id, string label, int node_index)
    {
        var url = $"get_node_image/{compression_job_id}/{label}/{node_index}";
        using var response = await _http.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var bytes = await response.Content.ReadAsByteArrayAsync();
        // Convert to base64 string for <img src="">
        string base64 = Convert.ToBase64String(bytes);
        return $"data:image/png;base64,{base64}";
    }


    public async Task<string> GetGraphJsonAsync(string compressionJobId, string label, int k)
    {
        try
        {
            var url = $"/get_graph_json/{compressionJobId}/{label}/{k}";
            using var response = await _http.PostAsync(url, null);
            response.EnsureSuccessStatusCode();
            var content = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(content);

            if (doc.RootElement.TryGetProperty("fig_json", out var figProp))
            {
                return figProp.GetString() ?? string.Empty;
            }

            return string.Empty; // No fig_json found case

        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error fetching graph JSON: {ex.Message}");
            throw;
        }
    }



    // --------------------- Summary -------------------------//
    public async Task<CompressionSummary?> GetCompressionSummaryFromMemoryAsync(string compression_job_id)
    {
        var url = $"/fetch_compression_summary_from_memory/{compression_job_id}";
        var response = await _http.GetFromJsonAsync<SummaryResponse>(url, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });
        return response?.Summary;
    }
    
    // --------------------- Container maintainence -------------------------//

    public async Task<string> DeleteContainerDataAsync(string compression_job_id)
    {
        var response = await _http.DeleteAsync($"/delete_container_data/{compression_job_id}");
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return json?["message"] ?? "Deleted successfully.";
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync();
            return $"Failed to delete: {error}";
        }
    }

    public async Task<string> DeleteAllContainerDataAsync()
    {
        var response = await _http.DeleteAsync($"/delete_all_container_data");
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return json?["message"] ?? "Deleted successfully.";
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync();
            return $"Failed to delete: {error}";
        }
    }

    public async Task<List<CompressionSummary>> LoadAllSummariesFromContainerAsync()
    {
        var summaries = await _http.GetFromJsonAsync<List<CompressionSummary>>("/summaries_from_container") 
                    ?? new List<CompressionSummary>();

        if (summaries.Count != 0)
        {
            // Sort by Timestamp descending (most recent first)
            summaries = summaries.OrderBy(s => s.DatasetName)
                            .ThenBy(s => s.K)
                            .ThenBy(s => s.Norm)
                            .ToList();
        }

        return summaries;
    }


    public async Task<SaveResponse> SaveCompressionAsync(string compression_job_id)
    {
        var response = await _http.PostAsJsonAsync(
            $"/save/{compression_job_id}", new { });
        var result = await response.Content.ReadFromJsonAsync<SaveResponse>();
        if (result is null)
        {
            throw new InvalidOperationException("Backend returned no data or invalid JSON");
        }
        return result;
    }



    public async Task<SaveResponse> HandleReplaceAsync(string compression_job_id, string duplicate_id)
    {
        Console.WriteLine("The duplicate id in api client is " + duplicate_id);
        Console.WriteLine("The compression id in api client is " + compression_job_id);
        var response = await _http.PostAsJsonAsync(
            $"/handle_replace_choice/{compression_job_id}/{duplicate_id}", new { });
        var result = await response.Content.ReadFromJsonAsync<SaveResponse>();
        if (result is null)
        {
            throw new InvalidOperationException("Backend returned no data or invalid JSON");
        }
        return result;

    }



    // --------------------- Maintain saved models -------------------------//

    public async Task<List<SavedModelInfo>> GetSavedModelInfoAsync()
    {
        var response = await _http.GetFromJsonAsync<List<SavedModelInfo>>("/get_models_info") 
                    ?? new List<SavedModelInfo>();

        // Sort by DatasetName in ascending (A-Z) order
        response = response
            .OrderBy(s => s.DatasetName)
            .ThenBy(s => s.K)
            .ThenBy(s => s.Kind)
            .ToList();

        return response;
    }


    public async Task<string> SaveModelAsync(int epoch, string trainId, SavedModelInfo info)
    {
        var response = await _http.PostAsJsonAsync($"/save_model/{trainId}/{epoch}", info);
        if (!response.IsSuccessStatusCode)
        {
            return $"Failed: {response.StatusCode}";
        }

        var result = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
        if (result != null && result.TryGetValue("message", out var msg))
        {
            return msg;
        }
        return "Unknown response from server.";
    }


    public async Task<string> DeleteCheckpointsAsync(string trainId)
    {
        var response = await _http.DeleteAsync($"/delete_checkpoints/{trainId}");
        if (!response.IsSuccessStatusCode)
            return $"Failed: {response.StatusCode}";

        var result = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
        return result?["message"] ?? "Unknown response";
    }

    public async Task<string> DeleteModelAsync(string modelId)
    {
        try
        {
            var response = await _http.DeleteAsync($"/delete_model/{modelId}");

            if (!response.IsSuccessStatusCode)
                return $"Delete failed: {response.StatusCode}";

            // Try to read JSON response
            var responseText = await response.Content.ReadAsStringAsync();

            try
            {
                var json = JsonSerializer.Deserialize<Dictionary<string, string>>(responseText);
                if (json != null && json.TryGetValue("message", out var msg))
                    return msg;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not parse JSON: {ex.Message}");
            }

            return string.IsNullOrWhiteSpace(responseText)
                ? "Model deleted."
                : responseText;
        }
        catch (Exception ex)
        {
            return $"Error deleting model: {ex.Message}";
        }
    }

    public async Task<string> DeleteAllModelsAsync()
    {
        try
        {
            var response = await _http.DeleteAsync("/delete_all_models");

            if (!response.IsSuccessStatusCode)
                return $"Failed: {response.StatusCode}";

            var result = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return result?["message"] ?? "All models deleted.";
        }
        catch (Exception ex)
        {
            return $"Error deleting all models: {ex.Message}";
        }
    }




    // --------------------- Pre-process Original Data -------------------------//
    public async Task<List<string>> GetDatasetNamesAsync()
    {
        var url = $"/get_all_dataset_names";
        var result = await _http.GetFromJsonAsync<List<string>>(url);
        return result ?? new List<string>();
    }


    public async Task<Dictionary<string, List<string>>> GetAllDatasetLabelsAsync()
    {
        try
        {
            var response = await _http.GetFromJsonAsync<Dictionary<string, List<string>>>("get_all_dataset_labels");
            return response ?? new Dictionary<string, List<string>>();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to fetch dataset labels: {ex.Message}");
            return new Dictionary<string, List<string>>();
        }
    }


    public async Task<string> DeleteDatasetAsync(string datasetName)
    {
        var url = $"/delete_dataset/{datasetName}";
        var response = await _http.DeleteAsync(url);
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return json?["message"] ?? "Deleted successfully.";
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync();
            return $"Failed to delete: {error}";
        }
    }


    public async Task<HttpResponseMessage> PostUserDatasetAsync(MultipartFormDataContent content)
    {
        return await _http.PostAsync("/upload", content);
    }




    // --------------------- Train History -------------------------//
    public async Task<List<TrainingRun>> GetHistoryAsync()
    {
        var response = await _http.GetStringAsync("/train_history");
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
        options.Converters.Add(new BaseTrainRequestConverter());
        var history = JsonSerializer.Deserialize<List<TrainingRun>>(response, options)
                    ?? new List<TrainingRun>();
        return history ;
    }




    public async Task<string> DeleteTrainingRunAsync(string train_id)
    {
        var response = await _http.DeleteAsync($"/delete_training_run/{train_id}");
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return json?["message"] ?? "Deleted successfully.";
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync();
            return $"Failed to delete: {error}";
        }
    }



    public async Task<string> DeleteAllHistoryAsync()
    {
        try
        {
            var response = await _http.DeleteAsync("/delete_all_history");
            if (!response.IsSuccessStatusCode)
                return $"Failed: {response.StatusCode}";

            var result = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            return result?["message"] ?? "All History deleted.";
        }
        catch (Exception ex)
        {
            return $"Error deleting all history: {ex.Message}";
        }
    }



    // --------------------- Train  -------------------------//
    public async Task CancelTrainingAsync(string train_id)
    {
        await _http.DeleteAsync($"/cancel_train/{train_id}");
    }



}
