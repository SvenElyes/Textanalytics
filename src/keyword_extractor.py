"""This module can extract keywords for either an entire chapter or all verses containing a character's name."""

import yake
import dataloader


def get_text_for_chapter(chapter, book_id):
    """Selects text relevant for chapter

    This function selects rows of a DataFrame (the bible data frame) for specified chapter and
    re-formats them as concatenated string for further use by the keyword extraction algorithm.

    :type chapter: int
    :type book_id: str
    :param chapter: the chapter for which text should be selected, e. g. 1
    :param book_id: the book in which this chapter is contained, e. g. Gen (Genesis)
    :return: the text values of the selected rows as a concatenated string
    :rtype: str
    """
    bible_df = dataloader.get_df_bible()
    ch_df = bible_df.loc[
        (bible_df["chapter"] == chapter) & (bible_df["book_id"] == book_id)
    ]
    ch_texts = ch_df["text"].tolist()
    text = " ".join(ch_texts)
    return text


# verses with character name


def get_text_for_character(character_name):
    """Selects text relevant for character

    This function selects rows of a DataFrame (the bible data frame) for specified character by choosing those verses
    in which the character's name appears. Then, it reduces or expands this relevant context to 20 words -
    the ten words before and after the character's name.
    It does so by looking up the previous/following verse by index in the bible data frame.

    :type character_name: str
    :param character_name: the character for which verses should be selected, e. g. "Jesus"
    :return: the text values of the selected rows as a concatenated string
    :rtype: str
    """
    bible_df = dataloader.get_df_bible()
    character_df = bible_df[bible_df["text"].str.contains(character_name)]
    character_verses = dict(
        zip(character_df.index, character_df.text)
    )  # FIXME in order to not iterate over df - good?
    character_texts = []

    for i, verse in character_verses.items():
        split_verse = verse.split(
            character_name
        )  # "He was Jesus and Jesus was good" -> ["He was", "and", "was good"]
        for index, verse_part in enumerate(split_verse[:-1]):
            context_before = character_name.join(
                split_verse[: index + 1]
            )  # index=1: "Jesus".join(["He was", "and"])
            context_size_before = len(context_before.split(" "))
            context_after = character_name.join(
                split_verse[index + 1 :]
            )  # index=0: "Jesus".join(["and", "was good"])
            context_size_after = len(context_after.split(" "))

            added_verses = 0
            while context_size_before < 10:
                added_verses += 1
                if i - added_verses > 0:
                    context_before = (
                        bible_df.loc[i - added_verses, "text"] + " " + context_before
                    )
                    context_size_before = len(context_before.split(" "))
                else:
                    break

            added_verses = 0
            while context_size_after < 10:
                added_verses += 1
                if i + added_verses < len(bible_df.index):
                    context_after += " " + bible_df.loc[i + added_verses, "text"]
                    context_size_after = len(context_after.split(" "))
                else:
                    break

            if context_size_before > 10:
                context_before = " ".join(context_before.split(" ")[:10])
            if context_size_after > 10:
                context_after = " ".join(context_after.split(" ")[:10])

            whole_context = context_before + " " + character_name + " " + context_after
            character_texts.append(whole_context)

    text = " ".join(character_texts)
    return text


def get_keywords(text):  # TODO: experiment to find best parameters
    """Extracts keywords from given text

    Extracts keywords from given text using keyword extraction algorithm YAKE. Currently uses basic parameters
    for algorithm, which can be optimized. For explanation of parameters, see YAKE documentation.

    :type text: str
    :param text: to be processed text, most likely created by get_text_for_character() or get_text_for_chapter()
    :return: extracted keywords as tuple including confidence as float, e.g. ('keyword', 0.042)
    :rtype: tuple
    """
    max_ngram_size = 1
    deduplication_threshold = 0.9
    deduplication_algo = "eqm"
    window_size = 1
    num_of_keywords = 20
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=window_size,
        top=num_of_keywords,
        features=None,
    )  # FIXME stopwords don't seem to be working
    keywords = kw_extractor.extract_keywords(text)
    return keywords


if __name__ == "__main__":
    chapter1_genesis_text = get_text_for_chapter(1, "Gen")
    chapter1_genesis_keywords = get_keywords(chapter1_genesis_text)
    print("Keywords for Genesis, chapter 1:\n")
    for kw in chapter1_genesis_keywords:
        print(kw)

    jesus_text = get_text_for_character("Jesus")
    print(jesus_text[:5000])
    jesus_keywords = get_keywords(jesus_text)
    print("\nKeywords for Jesus:\n")
    for kw in jesus_keywords:
        print(kw)
