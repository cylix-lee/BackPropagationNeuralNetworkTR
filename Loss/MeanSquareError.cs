using System.Diagnostics;

namespace BackPropagationNeuralNetworkTR.Loss;

class MeanSquareError : ILossFunction<IEnumerable<double>, double>
{
    public double Loss(IEnumerable<double> groundTruth, IEnumerable<double> output)
    {
        Debug.Assert(groundTruth.Count() == output.Count());

        var loss = 0.0;
        for (var i = 0; i < groundTruth.Count(); i++)
        {
            loss += 0.5 * Math.Pow(groundTruth.ElementAt(i) - output.ElementAt(i), 2);
        }
        return loss;
    }
}
