using CompressoApp.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using SharpCompress.Archives;


namespace CompressoApp.Components.Pages;

public partial class DatasetContainer : ComponentBase
{
    [Inject] private ApiClient Api { get; set; } = default!;
    [Inject] private DatasetInfoService DatasetInfoManager { get; set; } = default!;

    private List<string>? datasetNames;
    bool showUpload = false;
    bool isUploading = false;
    string uploadMsg ="";
    private List<string> defaultDatasets = new List<string> { "mnist", "cifar10", "cifar100", "svhn" };

    protected override async Task OnInitializedAsync()
    {
        await LoadDatasetNamesAsync();
        await DatasetInfoManager.RefreshFromBackendAsync();
    }

    private async Task<bool> LoadDatasetNamesAsync()
    {
        // Load all dataset names from the container
        var latest = await Api.GetDatasetNamesAsync();

        // Normalize: sort alphabetically for consistent comparison
        latest = latest?.OrderBy(name => name).ToList();

        // If we have no previous list or contents differ, update and return true
        if (datasetNames == null ||
            !datasetNames.OrderBy(name => name).SequenceEqual(latest!))
        {
            datasetNames = latest;
            return true;
        }

        return false;
    }

    private async Task ShowMessageAsync(string message, bool autoClear = true, int delayMs = 10000)
    {
        uploadMsg = message;
        StateHasChanged();

        if (autoClear)
        {
            await Task.Delay(delayMs);
            uploadMsg = string.Empty;
            await InvokeAsync(StateHasChanged);
        }
    }


    private async Task HandleDelete(string datasetName)
    {
        var resultMessage = await Api.DeleteDatasetAsync(datasetName);
        await LoadDatasetNamesAsync();
        await InvokeAsync(StateHasChanged);
    }

    private void ShowUploadDialog() => showUpload = !showUpload;

    private async Task OnInputFileChange(InputFileChangeEventArgs e)
    {
        string msg = "";
        isUploading = true;
        showUpload = false;
        var file = e.File;
        if (file == null)
        {
            msg = "No file selected.";
            return;
        }

        var allowedExtensions = new[] { ".zip", ".tar", ".tar.gz", ".tgz" }; //
        var fileExt = Path.GetExtension(file.Name).ToLower();

        // Handle multi-extension formats
        if (file.Name.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase) ||
            file.Name.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase))
        {
            fileExt = ".tar.gz";
        }

        if (!allowedExtensions.Contains(fileExt))
        {
            msg = $"Unsupported file type. Please upload one of: {string.Join(", ", allowedExtensions)}";
            return;
        }

        var tmpPath = Path.Combine(Path.GetTempPath(), file.Name);

        await using (var stream = File.Create(tmpPath))
        // maximum, 2 G
        await file.OpenReadStream(maxAllowedSize: 2_000_000_000).CopyToAsync(stream);

        var tmpUnpackFolder = Path.Combine(Path.GetTempPath(), "unPack_"+Path.GetFileNameWithoutExtension(file.Name));
        Directory.CreateDirectory(tmpUnpackFolder);

        string datasetName;
        if (file.Name.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase))
            datasetName = Path.GetFileNameWithoutExtension(Path.GetFileNameWithoutExtension(file.Name)); // removes .gz then .tar
        else
            datasetName = Path.GetFileNameWithoutExtension(file.Name);

        try
        {
            // --- Extract depending on file type ---
            if (fileExt == ".zip")
            {
                System.IO.Compression.ZipFile.ExtractToDirectory(tmpPath, tmpUnpackFolder);
            }
            else if (fileExt == ".tar" || fileExt == ".tar.gz") //|| fileExt == ".tgz"
            {
                string tarPath = tmpPath;

                // If gzipped, decompress first
                if (fileExt == ".tar.gz") // || fileExt == ".tgz"
                {
                    tarPath = Path.ChangeExtension(tmpPath, ".tar");
                    Console.WriteLine($"Decompressing GZip archive to temporary TAR: {tarPath}");

                    using (var input = File.OpenRead(tmpPath))
                    using (var gzip = new System.IO.Compression.GZipStream(input, System.IO.Compression.CompressionMode.Decompress))
                    using (var output = File.Create(tarPath))
                    {
                        await gzip.CopyToAsync(output);
                    }
                }

                // Now open the (seekable) TAR file
                using (var archive = SharpCompress.Archives.Tar.TarArchive.Open(tarPath))
                {
                    foreach (var entry in archive.Entries.Where(e => !e.IsDirectory))
                    {
                        entry.WriteToDirectory(tmpUnpackFolder, new SharpCompress.Common.ExtractionOptions
                        {
                            ExtractFullPath = true,
                            Overwrite = true
                        });
                    }
                }

                // Cleanup temp .tar if created
                if (tarPath != tmpPath && File.Exists(tarPath))
                    File.Delete(tarPath);
            }


            // --- Handle nested TAR (e.g., pseudo_dataset.tar inside) ---
            // --- Handle nested TAR (file or folder layer) ---
            var innerTarFile = Directory.GetFiles(tmpUnpackFolder, "*.tar", SearchOption.AllDirectories).FirstOrDefault();
            var innerTarFolder = Directory.GetDirectories(tmpUnpackFolder, "*.tar", SearchOption.AllDirectories).FirstOrDefault();

            if (innerTarFile != null)
            {
                Console.WriteLine($"Found nested TAR file: {innerTarFile}");
                using (var innerArchive = SharpCompress.Archives.Tar.TarArchive.Open(innerTarFile))
                {
                    foreach (var entry in innerArchive.Entries.Where(e => !e.IsDirectory))
                    {
                        entry.WriteToDirectory(tmpUnpackFolder, new SharpCompress.Common.ExtractionOptions
                        {
                            ExtractFullPath = true,
                            Overwrite = true
                        });
                    }
                }
                File.Delete(innerTarFile);
                Console.WriteLine($"Extracted and removed nested TAR file: {innerTarFile}");
            }
            else if (innerTarFolder != null)
            {
                Console.WriteLine($"Found TAR folder layer: {innerTarFolder}");
                foreach (var dir in Directory.GetDirectories(innerTarFolder))
                {
                    var dest = Path.Combine(tmpUnpackFolder, Path.GetFileName(dir));
                    if (Directory.Exists(dest))
                        Directory.Delete(dest, true);
                    Directory.Move(dir, dest);
                }
                foreach (var _file in Directory.GetFiles(innerTarFolder))
                {
                    var dest = Path.Combine(tmpUnpackFolder, Path.GetFileName(_file));
                    File.Move(_file, dest, overwrite: true);
                }
                Directory.Delete(innerTarFolder, true);
                Console.WriteLine($"Flattened TAR folder layer: {innerTarFolder}");
            }

            // --- Flatten redundant folder if archive contains same-name subdir ---
            var subdirs = Directory.GetDirectories(tmpUnpackFolder, "*", SearchOption.TopDirectoryOnly);
            if (subdirs.Length == 1)
            {
                var innerDir = subdirs[0];
                var innerName = Path.GetFileName(innerDir).Trim().ToLowerInvariant();
                var outerName = Path.GetFileName(tmpUnpackFolder).Trim().ToLowerInvariant();

                if (innerName == outerName)
                {
                    Console.WriteLine($"Detected same-name nested folder: {innerDir}");
                    foreach (var dir in Directory.GetDirectories(innerDir))
                    {
                        var dest = Path.Combine(tmpUnpackFolder, Path.GetFileName(dir));
                        if (Directory.Exists(dest))
                            Directory.Delete(dest, true);
                        Directory.Move(dir, dest);
                    }

                    foreach (var _file in Directory.GetFiles(innerDir))
                    {
                        var dest = Path.Combine(tmpUnpackFolder, Path.GetFileName(_file));
                        File.Move(_file, dest, overwrite: true);
                    }

                    Directory.Delete(innerDir, true);
                    Console.WriteLine($"Flattened redundant folder layer");
                }
            }

            FlattenAndClean(tmpUnpackFolder, Path.GetFileNameWithoutExtension(file.Name));

            // --- Validate dataset structure ---
            bool valid = ValidateDatasetStructure(tmpUnpackFolder);
            if (!valid)
            {
                Directory.Delete(tmpUnpackFolder, true);
                msg = "Invalid dataset format. Each class must be a folder containing images.";
                return;
            }



            // --- All checks passed — now compress again
            string compressedPath = Path.Combine(Path.GetTempPath(), $"{datasetName}.zip");
            if (File.Exists(compressedPath)) File.Delete(compressedPath);
            System.IO.Compression.ZipFile.CreateFromDirectory(tmpUnpackFolder, compressedPath, System.IO.Compression.CompressionLevel.Optimal, includeBaseDirectory: false);

            Console.WriteLine($"Recompressed validated dataset: {compressedPath}");

            // Send compressed file to backend via HTTP
            using var content = new MultipartFormDataContent();
            await using var fileStream = File.OpenRead(compressedPath);
            var fileContent = new StreamContent(fileStream);
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/zip");
            content.Add(fileContent, "file", $"{datasetName}.zip");

            var response = await Api.PostUserDatasetAsync(content); // new method in ApiClient

            if (response.IsSuccessStatusCode)
            {
                msg = "Upload successful!";
            }
            else
            {
                msg = $"Upload failed: {response.ReasonPhrase}";
            }

            msg = "Upload successful!";
        }
        catch (Exception ex)
        {
            msg = $"Upload failed: {ex.Message}";
        }

        finally
        {
            try
            {
                // Delete temporary upload file
                if (File.Exists(tmpPath))
                {
                    File.Delete(tmpPath);
                    Console.WriteLine($"Deleted temporary file: {tmpPath}");
                }

                // Delete temporary unpack folder
                if (Directory.Exists(tmpUnpackFolder))
                {
                    Directory.Delete(tmpUnpackFolder, true);
                    Console.WriteLine($"Deleted temporary unpack folder: {tmpUnpackFolder}");
                }

                // Delete recompressed dataset zip
                var compressedPath = Path.Combine(Path.GetTempPath(), $"{datasetName}.zip");
                if (File.Exists(compressedPath))
                {
                    File.Delete(compressedPath);
                    Console.WriteLine($"Deleted recompressed file: {compressedPath}");
                }
            }
            catch (Exception cleanupEx)
            {
                Console.WriteLine($"Cleanup error: {cleanupEx.Message}");
            }

            await LoadDatasetNamesAsync();
            showUpload = false;
            isUploading = false;
            StateHasChanged();
            await ShowMessageAsync(msg);
        }
    }







    private void FlattenAndClean(string datasetFolder, string archiveBaseName)
    {
        // Detect nested folder (e.g., dataset/dataset/)
        var innerDir = Path.Combine(datasetFolder, archiveBaseName);
        if (Directory.Exists(innerDir))
        {
            Console.WriteLine($"Detected nested folder: {innerDir}");

            // Move all subfolders up
            foreach (var dir in Directory.GetDirectories(innerDir))
            {
                var dest = Path.Combine(datasetFolder, Path.GetFileName(dir));
                if (Directory.Exists(dest))
                    Directory.Delete(dest, true);
                Directory.Move(dir, dest);
            }

            // Move all files up
            foreach (var file in Directory.GetFiles(innerDir))
            {
                var dest = Path.Combine(datasetFolder, Path.GetFileName(file));
                File.Move(file, dest, overwrite: true);
            }

            Directory.Delete(innerDir, true);
            Console.WriteLine($"Flattened nested folder structure");
        }

        // Clean up macOS junk
        // Add PaxHeader and other junk
        var ignoreDirs = new[] { "__MACOSX", "PaxHeader", ".DS_Store" };

        foreach (var junkDir in Directory.GetDirectories(datasetFolder, "*", SearchOption.AllDirectories)
                .Where(d => ignoreDirs.Contains(Path.GetFileName(d))))
        {
            Console.WriteLine($"Removing junk folder: {junkDir}");
            Directory.Delete(junkDir, true);
        }

        foreach (var junk in Directory.GetFiles(datasetFolder, ".DS_Store", SearchOption.AllDirectories))
        {
            Console.WriteLine($"Removing junk file: {junk}");
            File.Delete(junk);
        }

        foreach (var junk in Directory.GetFiles(datasetFolder, "._*", SearchOption.AllDirectories))
        {
            Console.WriteLine($"Removing AppleDouble file: {junk}");
            File.Delete(junk);
        }
    }



    bool ValidateDatasetStructure(string path)
    {
        //"PaxHeader",
        //var ignoreFolders = new[] { "__MACOSX",  ".DS_Store", "PaxHeader" };

        bool FolderHasImages(string folder)
        {
            // Skip junk
            var folderName = Path.GetFileName(folder);

            // Check for images directly in this folder
            var images = Directory.GetFiles(folder, "*.*")
                .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                        || f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                        || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
                .ToList();

            if (images.Any()) return true;

            // Otherwise, check subfolders recursively
            var subfolders = Directory.GetDirectories(folder);
            if (subfolders.Length == 0)
            {
                Console.WriteLine($"No images found in class folder: {folder}");
                return false;
            }

            return subfolders.All(FolderHasImages);
        }

        return FolderHasImages(path);
    }


}
     








