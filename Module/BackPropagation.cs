using System.Diagnostics;
using System.Text.Json;
using BackPropagationNeuralNetworkTR.Activation;

namespace BackPropagationNeuralNetworkTR.Module;

class BackPropagation : ILearnableModule<double, double>
{
    public static BackPropagation Load(string path)
    {
        using var file = File.OpenRead(path);
        using var reader = new StreamReader(file);
        return JsonSerializer.Deserialize<BackPropagation>(reader.ReadToEnd())!;
    }

    public int InputCount { get; }
    public int HiddenCount { get; }
    public int OutputCount { get; }
    public double LearningRate { get; }
    public IActivationFunction<double, double> ActivationFunction { get; }

    double[] input;
    readonly double[] hidden;
    readonly double[] output;
    readonly double[,] inputHiddenWeights;
    readonly double[,] hiddenOutputWeights;
    readonly double[] hiddenThresholds;
    readonly double[] outputThresholds;

    public BackPropagation(int inputCount, int hiddenCount, int outputCount, double learningRate,
                           IActivationFunction<double, double> activationFunction)
    {
        InputCount = inputCount;
        HiddenCount = hiddenCount;
        OutputCount = outputCount;
        LearningRate = learningRate;
        ActivationFunction = activationFunction;
        input = new double[inputCount];
        hidden = new double[hiddenCount];
        output = new double[outputCount];

        // Initializing weights.
        inputHiddenWeights = new double[inputCount, hiddenCount].RandomInitialize();
        hiddenOutputWeights = new double[hiddenCount, outputCount].RandomInitialize();
        hiddenThresholds = new double[hiddenCount].RandomInitialize();
        outputThresholds = new double[outputCount].RandomInitialize();
    }

    public double[] Forward(double[] input)
    {
        Debug.Assert(input.Length == InputCount);
        this.input = input;
        Array.Fill(hidden, 0);
        Array.Fill(output, 0);

        // Input -> Activated Hidden
        for (var j = 0; j < HiddenCount; j++)
        {
            for (var i = 0; i < InputCount; i++)
            {
                hidden[j] += input[i] * inputHiddenWeights[i, j];
            }
            hidden[j] = ActivationFunction.Activate(hidden[j] + hiddenThresholds[j]);
        }

        // Activated Hidden -> Activated Output
        for (var k = 0; k < OutputCount; k++)
        {
            for (var j = 0; j < HiddenCount; j++)
            {
                output[k] += hidden[j] * hiddenOutputWeights[j, k];
            }
            output[k] = ActivationFunction.Activate(output[k] + outputThresholds[k]);
        }
        return output;
    }

    public void Backward(double[] groundTruth)
    {
        Debug.Assert(groundTruth.Length == OutputCount);
        // Calculate errors.
        var hiddenOutputError = new double[OutputCount];
        for (var k = 0; k < OutputCount; k++)
        {
            hiddenOutputError[k] = (groundTruth[k] - output[k]) * ActivationFunction.Derivative(output[k]);
        }

        var inputHiddenError = new double[HiddenCount];
        for (var j = 0; j < HiddenCount; j++)
        {
            for (var k = 0; k < OutputCount; k++)
            {
                inputHiddenError[j] += hiddenOutputError[k] * hiddenOutputWeights[j, k];
            }
            inputHiddenError[j] *= ActivationFunction.Derivative(hidden[j]);
        }

        // Adjust parameters and thresholds.
        for (var k = 0; k < OutputCount; k++)
        {
            outputThresholds[k] += LearningRate * hiddenOutputError[k];
            for (var j = 0; j < HiddenCount; j++)
            {
                hiddenOutputWeights[j, k] += LearningRate * hiddenOutputError[k] * hidden[j];
            }
        }
        for (var j = 0; j < HiddenCount; j++)
        {
            hiddenThresholds[j] += LearningRate * inputHiddenError[j];
            for (var i = 0; i < InputCount; i++)
            {
                inputHiddenWeights[i, j] += LearningRate * inputHiddenError[j] * input[i];
            }
        }
    }

    public void Save(string path)
    {
        using var file = File.Create(path);
        using var writer = new StreamWriter(file);
        writer.Write(JsonSerializer.Serialize(this));
    }
}

static class RandomInitializationExtension
{
    public static T RandomInitialize<T>(this T sequence) where T : IList<double>
    {
        for (var i = 0; i < sequence.Count; i++)
        {
            sequence[i] = Random.Shared.NextDouble() / 1000;
        }
        return sequence;
    }

    public static double[,] RandomInitialize(this double[,] matrix)
    {
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i, j] = Random.Shared.NextDouble() / 1000;
            }
        }
        return matrix;
    }
}