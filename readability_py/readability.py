def main():
    # Prompt the user for some text
    text = input("Text: ")

    # Count letters, words, and sentences
    letters = count_letters(text)
    words = count_words(text)
    sentences = count_sentences(text)

    # L = average number of letters per 100 words
    # S = average number of sentences per 100 words
    L = letters / words * 100
    S = sentences / words * 100

    # Coleman-Liau index
    index = 0.0588 * L - 0.296 * S - 15.8

    # Round to the nearest whole number
    grade = round(index)

    # Print the grade level
    if grade < 1:
        print("Before Grade 1")
    elif grade >= 16:
        print("Grade 16+")
    else:
        print(f"Grade {grade}")


def count_letters(text):
    count = 0
    for char in text:
        if char.isalpha():
            count += 1
    return count


def count_words(text):
    return len(text.split())


def count_sentences(text):
    count = 0
    for char in text:
        if char in ".!?":
            count += 1
    return count


main()
