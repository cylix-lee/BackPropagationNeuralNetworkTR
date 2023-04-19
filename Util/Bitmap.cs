using System.Runtime.InteropServices;

namespace BackPropagationNeuralNetworkTR.Util;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
readonly struct BitmapFileHeader
{
    public readonly short FileType;
    public readonly uint FileSize;
    public readonly short Reserved1;
    public readonly short Reserved2;
    public readonly uint Offbits;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
readonly struct BitmapInformationHeader
{
    public readonly uint InformationHeaderSize;
    public readonly uint Width;
    public readonly uint Height;
    public readonly short Planes;
    public readonly short BitsPerPixel;
    public readonly uint Compression;
    public readonly uint ImageSize;
    public readonly int HorizontalResolution;
    public readonly int VerticalResolution;
    public readonly uint UsedColors;
    public readonly uint ImportantColors;
}

sealed class Bitmap
{
    public static byte[] ReadImageData(string path) => new Bitmap(path).ImageData;
    public static double[] ReadNormalizedImageData(string path) => new Bitmap(path).NormalizedImageData;

    public BitmapFileHeader FileHeader { get; }
    public BitmapInformationHeader InformationHeader { get; }
    public byte[] ImageData { get; }
    public double[] NormalizedImageData { get; }

    public Bitmap(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(path);

        using var imageFile = File.OpenRead(path);
        using var binaryReader = new BinaryReader(imageFile);

        FileHeader = binaryReader.ReadStructure<BitmapFileHeader>();
        InformationHeader = binaryReader.ReadStructure<BitmapInformationHeader>();

        var headerSize = Marshal.SizeOf<BitmapFileHeader>() + Marshal.SizeOf<BitmapInformationHeader>();
        if (FileHeader.Offbits > headerSize)
            binaryReader.ReadBytes((int)(FileHeader.Offbits - headerSize));
        ImageData = binaryReader.ReadBytes((int)(FileHeader.FileSize - FileHeader.Offbits));
        NormalizedImageData = new double[ImageData.Length];
        for (var i = 0; i < ImageData.Length; i++)
        {
            NormalizedImageData[i] = ImageData[i] / 255.0;
        }
    }
}

static class MarshalExtensions
{
    static T ToStructure<T>(this byte[] bytes) where T : struct
    {
        var size = Marshal.SizeOf<T>();
        if (size > bytes.Length)
            throw new ArgumentException($"Expected {size} bytes, got {bytes.Length} bytes.");

        var pointer = Marshal.AllocHGlobal(size);
        Marshal.Copy(bytes, 0, pointer, size);
        var target = Marshal.PtrToStructure<T>(pointer);
        Marshal.FreeHGlobal(pointer);
        return target;
    }

    public static T ReadStructure<T>(this BinaryReader reader) where T : struct
        => reader.ReadBytes(Marshal.SizeOf<T>()).ToStructure<T>();
}