namespace BackPropagationNeuralNetworkTR.Util;

static class Log
{
    record Tag(string Text, ConsoleColor Color);

    static readonly Tag OkTag = new(nameof(Ok) + "      ", ConsoleColor.Green);
    static readonly Tag InfoTag = new(nameof(Info) + "    ", ConsoleColor.Blue);
    static readonly Tag MatchTag = new(nameof(Match) + "   ", ConsoleColor.Green);
    static readonly Tag MismatchTag = new(nameof(Mismatch), ConsoleColor.Red);

    static void LogWithTag(Tag tag, string message)
    {
        var originalColor = Console.ForegroundColor;
        {
            Console.ForegroundColor = tag.Color;
            Console.Write(tag.Text);
        }
        Console.ForegroundColor = originalColor;
        Console.WriteLine($" {message}");
    }

    public static void Ok(string message) => LogWithTag(OkTag, message);
    public static void Info(string message) => LogWithTag(InfoTag, message);
    public static void Match(string message) => LogWithTag(MatchTag, message);
    public static void Mismatch(string message) => LogWithTag(MismatchTag, message);
}
