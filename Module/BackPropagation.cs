using System.Diagnostics;
using System.Text.Json;
using BackPropagationNeuralNetworkTR.Activation;
using BackPropagationNeuralNetworkTR.Loss;

namespace BackPropagationNeuralNetworkTR.Module;

class BackPropagation : ILearnableModule<byte, double>
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
    public ILossFunction<IEnumerable<double>, double> LossFunction { get; }

    byte[] input;
    readonly double[] hidden;
    readonly double[] output;
    readonly double[,] inputHiddenWeights;
    readonly double[,] hiddenOutputWeights;
    readonly double[] hiddenThresholds;
    readonly double[] outputThresholds;

    public BackPropagation(int inputCount, int hiddenCount, int outputCount, double learningRate,
                           IActivationFunction<double, double> activationFunction,
                           ILossFunction<IEnumerable<double>, double> lossFunction)
    {
        InputCount = inputCount;
        HiddenCount = hiddenCount;
        OutputCount = outputCount;
        LearningRate = learningRate;
        ActivationFunction = activationFunction;
        LossFunction = lossFunction;
        input = new byte[inputCount];
        hidden = new double[hiddenCount];
        output = new double[outputCount];

        // Initializing weights.
        inputHiddenWeights = new double[inputCount, hiddenCount].RandomInitialize();
        hiddenOutputWeights = new double[hiddenCount, outputCount].RandomInitialize();
        hiddenThresholds = new double[hiddenCount].RandomInitialize();
        outputThresholds = new double[outputCount].RandomInitialize();
    }

    public double[] Forward(byte[] input)
    {
        Debug.Assert(input.Length == InputCount);
        this.input = input;

        // Input -> Hidden
        for (var i = 0; i < InputCount; i++)
        {
            var inputElement = this.input[i];
            for (var j = 0; j < HiddenCount; j++)
            {
                hidden[j] += inputElement * inputHiddenWeights[i, j];
            }
        }

        // Activated Hidden -> Output
        for (var i = 0; i < HiddenCount; i++)
        {
            hidden[i] = ActivationFunction.Activate(hidden[i] + hiddenThresholds[i]);
            for (var j = 0; j < OutputCount; j++)
            {
                output[j] = hiddenOutputWeights[i, j] * hidden[i];
            }
        }

        // Activated Output returns.
        for (var i = 0; i < OutputCount; i++)
        {
            output[i] = ActivationFunction.Activate(output[i] + outputThresholds[i]);
        }
        return output;
    }

    public void Backward(double[] groundTruth)
    {
        // Calculate errors.
        var hiddenOutputError = new double[OutputCount];
        for (var i = 0; i < OutputCount; i++)
        {
            hiddenOutputError[i] = (groundTruth[i] - output[i]) * ActivationFunction.Derivative(output[i]);
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
        for (var j = 0; j < HiddenCount; j++)
        {
            for (var k = 0; k < OutputCount; k++)
            {
                hiddenOutputWeights[j, k] += LearningRate * hiddenOutputError[k] * hidden[j];
            }
        }
        for (var i = 0; i < InputCount; i++)
        {
            for (var j = 0; j < HiddenCount; j++)
            {
                inputHiddenWeights[i, j] += LearningRate * inputHiddenError[j] * input[i];
            }
        }
        for (var k = 0; k < OutputCount; k++) outputThresholds[k] += LearningRate * hiddenOutputError[k];
        for (var j = 0; j < HiddenCount; j++) hiddenThresholds[j] += LearningRate * inputHiddenError[j];
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
            sequence[i] = Random.Shared.NextDouble();
        }
        return sequence;
    }

    public static double[,] RandomInitialize(this double[,] matrix)
    {
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i, j] = Random.Shared.NextDouble();
            }
        }
        return matrix;
    }
}