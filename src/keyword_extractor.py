import yake
import dataloader

# as a test: keywords for an entire chapter

bible_df = dataloader.get_df_bible()
ch1_df = bible_df.loc[(bible_df['chapter'] == 1) & (bible_df['book_id'] == "Gen")]
ch1_texts = ch1_df['text'].tolist()
text = " ".join(ch1_texts)

max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = "eqm"
windowSize = 1
numOfKeywords = 10

kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram_size, dedupLim=deduplication_threshold,
                                     dedupFunc=deduplication_algo, windowsSize=windowSize,
                                     top=numOfKeywords, features=None)
keywords = kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)
