
namespace CompressoApp.Services;


public class DatasetInfoService
{
    private readonly ApiClient _api;

    public Dictionary<string, string> Descriptions { get; private set; } = new();
    public Dictionary<string, string> TmpInfo { get; private set; } = new();
    public Dictionary<string, List<string>> Labels { get; private set; } = new();

    public DatasetInfoService(ApiClient api)
    {
        _api = api;
        InitializeDefaults();
    }

    private void InitializeDefaults()
    {
        Descriptions = new()
        {
            ["mnist"] = "The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each sized 28x28 pixels. It is one of the most widely used benchmarks for image classification. Despite its simplicity, it remains highly useful for testing new machine learning methods and teaching fundamental concepts. MNIST serves as a standard starting point for evaluating image recognition models.",

            ["cifar10"] = "CIFAR-10 consists of 60,000 color images sized 32x32 pixels, divided evenly into 10 classes, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class contains 6,000 images, providing a balanced dataset. CIFAR-10 is widely used for developing and benchmarking deep learning models in small-scale image classification tasks.",

            ["cifar100"] = "CIFAR-100 is similar to CIFAR-10 but more fine-grained, containing 60,000 color images across 100 classes. Each class has only 600 images, making classification more challenging. The 100 classes are organized into 20 superclasses, allowing evaluation of models on both detailed and hierarchical categorization tasks. It is commonly used to test the robustness and generalization of image recognition models.",

            ["svhn"] = "The Street View House Numbers (SVHN) dataset contains over 600,000 digit images collected from real-world house numbers in Google Street View. The digits appear in natural scenes with varying backgrounds and lighting conditions, making it more complex than MNIST. SVHN is widely used to study digit recognition in noisy, real-world environments and to benchmark models under realistic conditions."
        };


        Labels = new()
        {
            ["mnist"] = new List<string> { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" },
            ["svhn"] = new List<string> { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" },
            //literally from datasets.classes, for consistency
            ["cifar10"] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            ["cifar100"] = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
                            "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar",
                            "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile",
                            "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
                            "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
                            "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
                            "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
                            "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
                            "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
                            "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe",
                            "whale", "willow_tree", "wolf", "woman", "worm"]

        };
    }

    // Load or refresh from backend (can be called at startup or after adding dataset)
    public async Task RefreshFromBackendAsync()
    {
        try
        {
            var dynamicLabels = await _api.GetAllDatasetLabelsAsync();
            MergeDynamicLabels(dynamicLabels);
            Console.WriteLine($"Merged {dynamicLabels.Count} dynamic datasets.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to refresh dataset labels: {ex.Message}");
        }
    }

    private void MergeDynamicLabels(Dictionary<string, List<string>> dynamicLabels)
    {
        foreach (var kvp in dynamicLabels)
        {
            Labels[kvp.Key] = kvp.Value; 
        }
    }
}

