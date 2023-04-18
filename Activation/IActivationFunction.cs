namespace BackPropagationNeuralNetworkTR.Activation;

interface IActivationFunction<TInput, TOutput>
{
    TOutput Activate(TInput x);
    TOutput Derivative(TInput x);
}
