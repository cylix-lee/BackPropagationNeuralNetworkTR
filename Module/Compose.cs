using System.Diagnostics;

namespace BackPropagationNeuralNetworkTR.Module;

class Compose<T> : IModule<T, T>
{
    public int InputCount { get; }
    public int OutputCount { get; }

    readonly List<IModule<T, T>> modules;

    public Compose(int inputCount, int outputCount, params IModule<T, T>[] modules)
    {
        InputCount = inputCount;
        OutputCount = outputCount;
        Debug.Assert(modules != null);
        Debug.Assert(modules.Length > 1);
        Debug.Assert(modules[0].InputCount == InputCount);
        for (var i = 1; i < modules.Length; i++)
        {
            Debug.Assert(modules[i - 1].OutputCount == modules[i].InputCount);
        }
        Debug.Assert(modules[^1].OutputCount == OutputCount);
        this.modules = modules.ToList();
    }

    public Compose(params IModule<T, T>[] modules) : this(modules[0].InputCount, modules[^1].OutputCount, modules) { }

    public T[] Forward(T[] input)
    {
        var periodicalResult = input;
        foreach (var module in modules)
        {
            periodicalResult = module.Forward(periodicalResult);
        }
        return periodicalResult;
    }
}
