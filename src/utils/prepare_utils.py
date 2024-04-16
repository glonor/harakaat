import re


def strip_diacritics(text):
    pattern = re.compile("[\u064b-\u0652]")

    stripped_text = pattern.sub("", text)

    return stripped_text


def truncate(example, max_len):
    """Truncate the diacratized text to fit within the max_len byte limit."""
    target_line = example["diacratized"]

    target_line = target_line.split()

    new_target_line = ""
    byte_count = 0

    # Iterate over each word
    for target_word in target_line:
        # Calculate the size of the resulting sentence if we add the next word
        next_byte_count = byte_count + len(target_word.encode("utf-8"))
        if new_target_line:
            next_byte_count += len(" ")

        # Check if adding the next word will exceed the byte limit
        if next_byte_count > max_len - 3:  # -3 for the special tokens
            break

        # Add preceding space only if it's not the first word
        if new_target_line:
            new_target_line += " "
        new_target_line += target_word

        byte_count = next_byte_count

    example["diacratized"] = new_target_line
    example["text"] = strip_diacritics(new_target_line)

    return example


def segment(batch, max_len):
    """Segment the diacratized text to fit within the max_len byte limit."""
    diacratized = []
    text = []
    for target_line in batch["diacratized"]:

        target_line = target_line.split()

        new_target_line = ""
        byte_count = 0

        # Iterate over each word
        for target_word in target_line:
            # Calculate the size of the resulting sentence if we add the next word
            next_byte_count = byte_count + len(target_word.encode("utf-8"))
            if new_target_line:
                next_byte_count += len(" ")

            # Check if adding the next word will exceed the byte limit
            if next_byte_count > max_len - 3:  # -3 for the special tokens
                diacratized.append(new_target_line)
                text.append(strip_diacritics(new_target_line))
                byte_count = 0
                new_target_line = ""
                continue

            # Add preceding space only if it's not the first word
            if new_target_line:
                new_target_line += " "
            new_target_line += target_word

            byte_count = next_byte_count

        diacratized.append(new_target_line)
        text.append(strip_diacritics(new_target_line))

    return {"new_diacratized": diacratized, "new_text": text}
