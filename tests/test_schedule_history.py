import copy
import unittest

from pydantic import ValidationError

from schedule_history import resolve_schedule_history


def decision(status, evidence, expression=None, timestamp=None, content='리뷰'):
    return {
        'status': status, 'timeExpression': None, 'evidence': evidence, 'dateExpression': expression,
        'extractedScheduleDate': timestamp, 'extractedScheduleContent': content,
    }


class ScheduleHistoryTest(unittest.TestCase):
    def setUp(self):
        self.first = decision('confirmed', '리뷰는 내일 오전 10시로 확정합니다.',
                              '내일', '2026-09-26T10:00:00')
        self.proposal = decision('tentative', '리뷰를 모레로 옮기면 어떨까요?')
        self.cancel = decision('cancelled', '리뷰 일정은 취소합니다.')
        self.changed = decision('confirmed', '리뷰는 모레 오전 11시로 다시 확정합니다.',
                                '모레', '2026-09-27T11:00:00')

    def source(self, *decisions):
        return '\n'.join(item['evidence'] for item in decisions)

    def test_latest_confirmation_replaces_previous_date_and_time(self):
        source = self.source(self.first, self.proposal, self.changed)
        decisions = [self.changed, self.first, self.proposal]
        original = copy.deepcopy(decisions)

        result = resolve_schedule_history(decisions, source)

        self.assertEqual(result, self.changed)
        self.assertEqual(decisions, original)

    def test_later_unconfirmed_change_keeps_existing_confirmation(self):
        self.assertEqual(
            resolve_schedule_history([self.first, self.proposal], self.source(self.first, self.proposal)),
            self.first,
        )

    def test_cancellation_removes_confirmation_even_if_array_order_is_reversed(self):
        self.assertIsNone(resolve_schedule_history(
            [self.cancel, self.first], self.source(self.first, self.cancel),
        ))

    def test_proposal_after_cancellation_does_not_restore_schedule(self):
        decisions = [self.first, self.cancel, self.proposal]
        self.assertIsNone(resolve_schedule_history(decisions, self.source(*decisions)))

    def test_confirmation_after_cancellation_restores_schedule(self):
        decisions = [self.first, self.cancel, self.changed]
        self.assertEqual(resolve_schedule_history(decisions, self.source(*decisions)), self.changed)

    def test_no_confirmation_returns_no_schedule(self):
        for decisions in [[], [self.proposal], [self.cancel], [self.proposal, self.cancel]]:
            with self.subTest(decisions=decisions):
                self.assertIsNone(resolve_schedule_history(decisions, self.source(*decisions)))

    def test_separate_event_histories_do_not_cancel_each_other(self):
        release = decision('confirmed', '배포는 모레 오후 2시로 확정합니다.',
                           '모레', '2026-09-27T14:00:00', content='배포')
        source = self.source(self.first, release, self.cancel)
        self.assertIsNone(resolve_schedule_history([self.first, self.cancel], source))
        self.assertEqual(resolve_schedule_history([release], source), release)

    def test_repeated_source_quote_requires_unambiguous_evidence(self):
        source = self.source(self.first, self.cancel, self.first)
        with self.assertRaisesRegex(ValueError, '같은 근거 발언이 원문에 반복'):
            resolve_schedule_history([self.first, self.cancel], source)

    def test_duplicate_or_overlapping_decision_spans_are_rejected(self):
        source = self.source(self.first, self.changed)
        broad_quote = {**self.first, 'evidence': source}
        for decisions in [[self.first, self.first], [broad_quote, self.changed]]:
            with self.subTest(decisions=decisions):
                with self.assertRaisesRegex(ValueError, '근거 구간이 겹쳐'):
                    resolve_schedule_history(decisions, source)

    def test_cancelled_decision_with_fabricated_evidence_cannot_remove_schedule(self):
        with self.assertRaisesRegex(ValidationError, '근거 발언이 회의 원문에 그대로 존재하지 않습니다'):
            resolve_schedule_history([self.first, self.cancel], self.source(self.first))


if __name__ == '__main__':
    unittest.main()
