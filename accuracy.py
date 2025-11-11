from utils.text_utils import get_cer, get_wer


def get_errors(a: list, b: list):
    """
    a: ground truth text (label)
    b: predicted text
    return (S, D, I) converting sequence a -> sequence b (lists of tokens/chars).
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution/match
            )
    i, j = m, n
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I

# before calling test_accuracy, we should make sure that label is not emppty string
def test_accuracy(predicted, label):
    S, D, I = get_errors(list(label), list(predicted))
    correct = max(0, len(label) - S - D) # takes into account insertion errors
    print(f"Label: {label}, predicted: {predicted}, S: {S}, D: {D}, I: {I}, Correct chars: {correct}/{max(len(label), len(predicted))}")

    cer = get_cer(predicted, label)
    wer = get_wer(predicted, label)
    print(f"CER: {cer:.6%}, WER: {wer:.6%}")

    char_accuracy = (correct / max(len(label), len(predicted))) * 100
    word_accuracy = 100.0 if char_accuracy == 100.0 else 0.0

    print(f"Character Accuracy: {char_accuracy:.2f}%, Word Accuracy: {word_accuracy:.2f}%\n")

    return char_accuracy, word_accuracy

# test_accuracy("hello", "hallo")
# test_accuracy("caat", "cat")
# test_accuracy("kitten", "sitting")
# test_accuracy("law", "lawn")
# test_accuracy("hello", "hello")
