using BackPropagationNeuralNetworkTR.Util;

namespace BackPropagationNeuralNetworkTR.Dataset;

class YaleDataset : IBatchedDataset<double[], int>
{
    public int ItemCount => SubjectCount * SampleCountPerSubject;
    public int SubjectCount => 15;
    public int SampleCountPerSubject => 11;

    public (double[], int) this[int index]
    {
        get
        {
            var subject = index / SampleCountPerSubject;
            var sample = index % SampleCountPerSubject;
            return this[subject, sample];
        }
    }
    public (double[], int) this[int subject, int sample] => (images[subject, sample], subject);

    readonly double[,][] images;

    public YaleDataset()
    {
        images = new double[SubjectCount, SampleCountPerSubject][];
        var format = "YALE/subject{0:D2}_{1:D}.bmp";
        for (var subject = 1; subject <= SubjectCount; subject++)
        {
            for (var sample = 1; sample <= SampleCountPerSubject; sample++)
            {
                var filename = string.Format(format, subject, sample);
                images[subject - 1, sample - 1] = Bitmap.ReadNormalizedImageData(filename)[..^2];
            }
        }
        Console.WriteLine("Loaded YALE image dataset.");
    }

    public (double[], int)[] GetBatch(int subject)
    {
        var batch = new (double[], int)[SampleCountPerSubject];
        for (var i = 0; i < SampleCountPerSubject; i++)
        {
            batch[i] = this[subject, i];
        }
        return batch;
    }
}
