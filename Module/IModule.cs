namespace BackPropagationNeuralNetworkTR.Module;

interface IModule<TInput, TOutput>
{
    int InputCount { get; }
    int OutputCount { get; }

    TOutput[] Forward(TInput[] input);
}

interface ILearnableModule<TInput, TOutput> : IModule<TInput, TOutput>
{
    void Backward(TOutput[] groundTruth);
}