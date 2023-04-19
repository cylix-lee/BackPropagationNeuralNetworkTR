namespace BackPropagationNeuralNetworkTR.Activation;

[Serializable]
class Sigmoid : IActivationFunction<double, double>
{
    public double Activate(double x) => 1 / (1 + Math.Exp(-x));
    public double Derivative(double x) => x * (1 - x);
}
