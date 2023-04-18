namespace BackPropagationNeuralNetworkTR.Dataset;

interface IDataset<TInput, TGroundTruth>
{
    int ItemCount { get; }
    (TInput, TGroundTruth) this[int index] { get; }
}

interface IBatchedDataset<TInput, TGroundTruth> : IDataset<TInput, TGroundTruth>
{
    int SubjectCount { get; }
    int SampleCountPerSubject { get; }
    (TInput, TGroundTruth) this[int subject, int sample] { get; }

    (TInput, TGroundTruth)[] GetBatch(int subject);
}

static class DatasetExtension
{
    public static (TInput, TGroundTruth)[]
        GetAllItems<TInput, TGroundTruth>(this IDataset<TInput, TGroundTruth> dataset)
    {
        var items = new (TInput, TGroundTruth)[dataset.ItemCount];
        for (var i = 0; i < dataset.ItemCount; i++)
        {
            items[i] = dataset[i];
        }
        return items;
    }
}

static class BatchedDatasetExtension
{
    public static (TInput, TGroundTruth)[][]
        GetAllBatches<TInput, TGroundTruth>(this IBatchedDataset<TInput, TGroundTruth> dataset)
    {
        var batches = new (TInput, TGroundTruth)[dataset.SubjectCount][];
        for (var i = 0; i < dataset.SubjectCount; i++)
        {
            batches[i] = dataset.GetBatch(i);
        }
        return batches;
    }

    public static ((TInput, TGroundTruth)[] TrainSet, (TInput, TGroundTruth)[] TestSet)
        Split<TInput, TGroundTruth>(this IBatchedDataset<TInput, TGroundTruth> dataset, int trainSetSize)
    {
        var testSetSize = dataset.SampleCountPerSubject - trainSetSize;
        var trainSet = new (TInput, TGroundTruth)[dataset.SubjectCount * trainSetSize];
        var testSet = new (TInput, TGroundTruth)[dataset.SubjectCount * testSetSize];

        var trainSetIndex = 0;
        var testSetIndex = 0;
        for (var i = 0; i < dataset.SubjectCount; i++)
        {
            for (var j = 0; j < dataset.SampleCountPerSubject; j++)
            {
                if (j < trainSetSize) trainSet[trainSetIndex++] = dataset[i, j];
                else testSet[testSetIndex++] = dataset[i, j];
            }
        }
        return (trainSet, testSet);
    }
}