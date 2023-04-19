using System.Runtime.InteropServices;

namespace BackPropagationNeuralNetworkTR.Util;

static class MarshalExtensions
{
    public static T ToStructure<T>(this byte[] bytes) where T : struct
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