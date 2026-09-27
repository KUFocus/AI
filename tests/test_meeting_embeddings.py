import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from meeting_embeddings import LocalMeetingEmbeddings


class MeetingEmbeddingsTests(unittest.TestCase):
    def make_embedder(self, *, tokens=3, batch_size=8):
        tokenizer = Mock()
        def tokenize(texts, **kwargs):
            mask = torch.ones(len(texts), tokens, dtype=torch.long)
            mask[:, -1] = 0
            return {'input_ids': torch.ones_like(mask), 'attention_mask': mask}
        tokenizer.side_effect = tokenize
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        def encode(**batch):
            count, length = batch['input_ids'].shape
            states = torch.zeros(count, length, 384)
            states[:, :-1, 0] = 3
            states[:, :-1, 1] = 4
            states[:, -1, 2] = 1000
            return SimpleNamespace(last_hidden_state=states)
        model.side_effect = encode
        return LocalMeetingEmbeddings(tokenizer, model, batch_size=batch_size)

    def test_padding_is_excluded_and_vectors_are_normalized(self):
        embedder = self.make_embedder()
        vector = embedder.embed_documents(['담당 업무를 확인한다.'])[0]
        self.assertEqual(len(vector), 384)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)
        self.assertEqual(vector[2], 0)

    def test_query_and_document_prefixes_differ_for_korean(self):
        embedder = self.make_embedder()
        embedder.embed_documents(['민수가 기록을 보관한다.'])
        self.assertEqual(embedder.tokenizer.call_args.args[0], ['passage: 민수가 기록을 보관한다.'])
        embedder.embed_query('누가 기록을 보관하나요?')
        self.assertEqual(embedder.tokenizer.call_args.args[0], ['query: 누가 기록을 보관하나요?'])
        self.assertFalse(embedder.tokenizer.call_args.kwargs['truncation'])

    def test_empty_documents_skip_model_and_blank_text_is_rejected(self):
        embedder = self.make_embedder()
        self.assertEqual(embedder.embed_documents([]), [])
        embedder.model.assert_not_called()
        for texts in [[''], ['   '], [None], '문자열']:
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                embedder.embed_documents(texts)
        with self.assertRaises(ValueError):
            embedder.embed_query(' ')
        embedder.model.assert_not_called()

    def test_token_limit_is_checked_before_inference(self):
        long = self.make_embedder(tokens=513)
        with self.assertRaisesRegex(ValueError, '먼저 분할'):
            long.embed_documents(['긴 원문'])
        long.model.assert_not_called()
        self.assertEqual(len(self.make_embedder(tokens=512).embed_query('질문')), 384)

    def test_batches_keep_order_and_size(self):
        embedder = self.make_embedder(batch_size=2)
        self.assertEqual(len(embedder.embed_documents(['가', '나', '다'])), 3)
        self.assertEqual([c.args[0] for c in embedder.tokenizer.call_args_list],
                         [['passage: 가', 'passage: 나'], ['passage: 다']])
        self.assertEqual(embedder.model.call_count, 2)

    def test_invalid_model_outputs_are_rejected(self):
        for states in [torch.zeros(1, 3, 384), torch.full((1, 3, 384), float('nan')), torch.ones(1, 3, 10)]:
            with self.subTest(shape=states.shape):
                embedder = self.make_embedder()
                embedder.model.side_effect = None
                embedder.model.return_value = SimpleNamespace(last_hidden_state=states)
                with self.assertRaises(ValueError):
                    embedder.embed_query('질문')

    def test_invalid_batch_size_is_rejected(self):
        for size in [0, -1, True, 1.5]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.make_embedder(batch_size=size)

    def test_missing_local_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '로컬 폴더'):
            LocalMeetingEmbeddings.from_local_directory('/없는/모델/폴더')


if __name__ == '__main__':
    unittest.main()
