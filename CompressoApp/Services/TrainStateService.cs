// using CompressoApp.Models;
// public class TrainStateService
// {
//     public string CurrentTrainId { get; set; } = "";
//     public string DefaultDataId { get; set; } = "";
//     public CompressionSummary? DefaultSummary { get; set; }
//     public string FinalDataId { get; set; } = "";
//     public CompressionSummary? FinalSummary { get; set; }


//     // Train settings
//     public string SelectedTrainingType { get; set; } = "";
//     public bool RequireFinalAdvAttackTest { get; set; } = false;


//     // Standard Training
//     public string StandardOptimizer { get; set; } = "SGD";
//     public int StandardItr { get; set; } = 10;
//     public double StandardLr { get; set; } = 0.01;

//     // Adversarial Training
//     public string AdvAttack { get; set; } = "PGD-linf";
//     public double AdvEps { get; set; } = 0.3;
//     public string AdvOptimizer { get; set; } = "Adam";
//     public int AdvItr { get; set; } = 10;
//     public double AdvLr { get; set; } = 0.01;
//     public double AdvAlpha { get; set; } = 0.01;



//     // training progress
//     public int ElapsedSeconds { get; set; } = 0;
//     public bool IsTraining { get; set; } = false;
//     public bool HasCompleted { get; set; } = false;
//     public bool IsPreparingForTraining { get; set; } = false;
//     public bool IsTerminating { get; set; } = false;
//     public bool HasTerminated { get; set; } = false;

//     // result
//     public record struct EpochMetrics(int Epoch, double TrainAcc, double TestAcc, double AdvAcc);

//     public List<EpochMetrics> EpochMetricsList { get; set; } = new List<EpochMetrics>();


// }
