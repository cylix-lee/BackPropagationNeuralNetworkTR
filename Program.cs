using BackPropagationNeuralNetworkTR.Activation;
using BackPropagationNeuralNetworkTR.Dataset;
using BackPropagationNeuralNetworkTR.Module;
using BackPropagationNeuralNetworkTR.Util;

namespace BackPropagationNeuralNetworkTR;

class Program
{
    const int Epoch = 7;
    const double LearningRate = 0.3;

    static void Main(string[] args)
    {
        var dataset = new YaleDataset();
        var (trainSet, testSet) = dataset.Split(5);
        var model = new BackPropagation(8000, 64, dataset.SubjectCount, LearningRate, new Sigmoid());
        // Training
        Console.WriteLine($"Start training model with hidden {model.HiddenCount} neurons, learning rate {model.LearningRate}");
        for (var epoch = 0; epoch < Epoch; epoch++)
        {
            var correct = 0;
            foreach (var (input, groundTruth) in trainSet)
            {
                var output = model.Forward(input);
                if (output.Argmax() == groundTruth)
                    correct++;
                model.Backward(groundTruth.ToOneHotEncoding(dataset.SubjectCount));
            }
            Log.Ok($"Epoch {epoch} training finished with accuracy {(double)correct / trainSet.Length}");
        }
        Console.WriteLine("Training finished. Saving model...");
        model.Save("BP.json");
        Console.WriteLine();

        // Test model accuracy
        Console.WriteLine("Start testing model accuracy...");
        var correctCases = 0;
        foreach (var (input, groundTruth) in testSet)
        {
            var output = model.Forward(input).Argmax();
            if (output == groundTruth)
            {
                Log.Match($"Expected {groundTruth}, got {output}");
                correctCases++;
            }
            else Log.Mismatch($"Expected {groundTruth}, got {output}");
        }
        Console.WriteLine($"Testing finished with model accuracy {(double)correctCases / testSet.Length}");
    }
}

static class OutputExtension
{
    public static int Argmax(this double[] sequence)
    {
        var index = 0;
        var maxValue = sequence.First();
        for (var i = 1; i < sequence.Length; i++)
        {
            var candidate = sequence.ElementAt(i);
            if (candidate.CompareTo(maxValue) > 0)
            {
                index = i;
                maxValue = candidate;
            }
        }
        return index;
    }

    public static double[] ToOneHotEncoding(this int n, int categories)
    {
        var encoding = new double[categories];
        encoding[n] = 1;
        return encoding;
    }
}