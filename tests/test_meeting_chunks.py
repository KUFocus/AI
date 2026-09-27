import unittest

from meeting_chunks import MeetingChunker


class MeetingChunksTests(unittest.TestCase):
    # 토큰화 방식에 의존하지 않고 접두사와 특수 토큰 비용을 포함하는 시험용 계산기다.
    @staticmethod
    def count(text):
        return len(text) + 4

    def assert_source_preserved(self, text, chunks, limit):
        cursor = 0
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.index, index)
            self.assertLessEqual(chunk.start, cursor)
            self.assertGreater(chunk.end, cursor)
            self.assertEqual(text[chunk.start:chunk.end], chunk.text)
            self.assertTrue(chunk.text.strip())
            self.assertEqual(chunk.token_count, self.count(chunk.text))
            self.assertLessEqual(chunk.token_count, limit)
            cursor = chunk.end
        self.assertEqual(cursor, len(text))

    def test_short_meeting_preserves_spaces_and_line_breaks(self):
        text = '  민수: 기록을 확인합니다.\r\n지연: 결과를 공유합니다.  \n'
        chunks = MeetingChunker(self.count).split(text)
        self.assertEqual(len(chunks), 1)
        self.assert_source_preserved(text, chunks, 512)

    def test_sentences_are_kept_together_when_they_fit(self):
        text = '첫 업무를 맡습니다. 다음 업무를 맡습니다. 마지막 보고입니다.'
        chunks = MeetingChunker(self.count, max_tokens=29, overlap_tokens=8).split(text)
        self.assertTrue(chunks[0].text.endswith('. '))
        self.assert_source_preserved(text, chunks, 29)

    def test_long_unpunctuated_turn_is_split_without_loss(self):
        text = '민수: ' + '가나다라마바사아자차' * 35 + ' 끝'
        chunks = MeetingChunker(self.count, max_tokens=40, overlap_tokens=12).split(text)
        self.assertGreater(len(chunks), 1)
        self.assert_source_preserved(text, chunks, 40)
        self.assertEqual(chunks, MeetingChunker(self.count, max_tokens=40, overlap_tokens=12).split(text))

    def test_unicode_offsets_are_character_positions(self):
        text = '🙋 민수: Cafe\u0301 자료를 검토합니다.\n한글 가 발언입니다.'
        chunks = MeetingChunker(self.count, max_tokens=22, overlap_tokens=8).split(text)
        self.assert_source_preserved(text, chunks, 22)

    def test_token_limit_includes_prefix_cost(self):
        text = '가' * 20
        self.assert_source_preserved(text, MeetingChunker(self.count, max_tokens=24, overlap_tokens=0).split(text), 24)
        chunks = MeetingChunker(self.count, max_tokens=23, overlap_tokens=0).split(text)
        self.assertGreater(len(chunks), 1)
        self.assert_source_preserved(text, chunks, 23)

    def test_invalid_inputs_and_limits_are_rejected(self):
        for text in ['', '  \n', None, 1]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                MeetingChunker(self.count).split(text)
        for limit in [0, 513, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MeetingChunker(self.count, max_tokens=limit)

    def test_stt_without_speaker_or_punctuation_has_overlap(self):
        text = '요청 기록을 먼저 확인하고 결과가 다르면 저장 상태도 비교해서 공유해주세요 ' * 8
        chunks = MeetingChunker(self.count, max_tokens=80, overlap_tokens=30).split(text)
        self.assert_source_preserved(text, chunks, 80)
        self.assertTrue(any(right.start < left.end for left, right in zip(chunks, chunks[1:])))

    def test_ocr_line_breaks_and_table_rows_preserve_source(self):
        text = '항목\t수량\t상태\n노트북\t2\t점검중\n이번 주까지 구성품을\n확인하고 결과를 공유합니다.\n' * 5
        chunks = MeetingChunker(self.count, max_tokens=80, overlap_tokens=25).split(text)
        self.assert_source_preserved(text, chunks, 80)

    def test_repeated_text_has_consistent_offsets(self):
        text = '가나다라마바사' * 20
        chunks = MeetingChunker(self.count, max_tokens=45, overlap_tokens=15).split(text)
        self.assert_source_preserved(text, chunks, 45)

    def test_invalid_overlap_is_rejected(self):
        for overlap in [-1, 512, True, 1.5]:
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                MeetingChunker(self.count, overlap_tokens=overlap)

    def test_impossible_single_character_does_not_loop(self):
        with self.assertRaisesRegex(ValueError, '한도'):
            MeetingChunker(self.count, max_tokens=4, overlap_tokens=0).split('가')


if __name__ == '__main__':
    unittest.main()
