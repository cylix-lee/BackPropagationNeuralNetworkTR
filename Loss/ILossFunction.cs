namespace BackPropagationNeuralNetworkTR.Loss;

interface ILossFunction<TData, TLoss>
{
    TLoss Loss(TData groundTruth, TData output);
}
