from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock


from src.keyword_extractor import (
    get_keywords,
    get_text_for_chapter,
    get_text_for_character,
)
import src.dataloader
import pandas as pd


class KeywordextracterTest(TestCase):
    maxDiff = None

    def test_get_text_for_chapter(self):
        with patch("src.dataloader.pd") as mock_pd:

            mock_pd.read_csv.return_value = pd.read_csv(
                "test/csv_test/test_bible_4_verse.csv"
            )
            # Test bible only contains the first 18 verses
            response = "In the beginning God created the heavens and the earth. And the earth was waste and void; and darkness was upon the face of the deep: and the Spirit of God moved upon the face of the waters"
            self.assertEqual(
                get_text_for_chapter(1, "Gen"),
                response,
            )

    def test_get_text_for_character(self):
        with patch("src.dataloader.pd") as mock_pd:
            mock_pd.read_csv.return_value = pd.read_csv(
                "test/csv_test/test_whole_bible.csv"
            )

            Silas_text = "Then it seemed good to the apostles and the elders, Silas , chief men among the brethren: and they wrote [thus] men that have hazarded their lives for the name of Silas , who themselves also shall tell you the same things And when they had read it, they rejoiced for the Silas , being themselves also prophets, exhorted the brethren with many And after they had spent some time [there], they were Silas  to abide there.] But Paul and Barnabas tarried in And there arose a sharp contention, so that they parted Silas , and went forth, being commended by the brethren to But when her masters saw that the hope of their Silas , and dragged them into the marketplace before the rulers, who, having received such a charge, cast them into the Silas  were praying and singing hymns unto God, and the And he called for lights and sprang in, and, trembling Silas , and brought them out and said, Sirs, what must And some of them were persuaded, and consorted with Paul Silas , and of the devout Greeks a great multitude, and And when they had taken security from Jason and the Silas  by night unto Beroea: who when they were come And then immediately the brethren sent forth Paul to go Silas  and Timothy abode there still. But they that conducted But they that conducted Paul brought him as far as Silas  and Timothy that they should come to him with And he reasoned in the synagogue every sabbath, and persuaded Silas  and Timothy came down from Macedonia, Paul was constrained"
            self.assertEqual(get_text_for_character("Silas"), Silas_text)

    def test_get_keywords(self):
        first4verses = "In the beginning God created the heavens and the earth. And the earth was waste and void; and darkness was upon the face of the deep: and the Spirit of God moved upon the face of the waters"

        self.assertIsInstance(get_keywords(first4verses), list)
        self.assertIsInstance(get_keywords(first4verses)[0], tuple)
