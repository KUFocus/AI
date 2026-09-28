import json
import unittest
from datetime import date, datetime, timedelta
from itertools import product
from unittest.mock import Mock

from schedule_tools import resolve_schedule_date, resolve_schedule_time
from summarization import MeetingSummarizer


PARTICLES = ['', '에', '에는', '에도', '까지', '까지는', '까지도', '까지로',
             '부터', '부터는', '부터도', '로', '로는', '로도']


class TemporalExpressionGrammarTest(unittest.TestCase):
    def test_particles_preserve_valid_clock_values_across_hours_and_notations(self):
        for hour, minute, particle in product(range(24), [0, 17, 59], PARTICLES):
            period = '오전' if hour < 12 else '오후'
            expressions = [f'{period} {hour % 12 or 12}시 {minute}분{particle}',
                           f'{hour:02}:{minute:02}{particle}']
            for expression in expressions:
                with self.subTest(expression=expression):
                    self.assertEqual(resolve_schedule_time.invoke({'expression': expression}),
                                     f'{hour:02}:{minute:02}:00')

    def test_particles_preserve_relative_dates_at_month_year_and_leap_boundaries(self):
        for reference, offset, particle in product(
            [date(2027, 12, 31), date(2028, 2, 28), date(2028, 2, 29), date(2029, 4, 30)],
            range(3), PARTICLES,
        ):
            expression = ['오늘', '내일', '모레'][offset] + particle
            with self.subTest(reference=reference, expression=expression):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': expression, 'meeting_date': reference.isoformat(),
                }), (reference + timedelta(days=offset)).isoformat())

    def test_absolute_dates_and_next_weekdays_share_particle_rules(self):
        for month, particle in product(range(1, 13), PARTICLES):
            for expression in [f'2029년 {month}월 16일{particle}', f'2029-{month:02}-16{particle}']:
                with self.subTest(expression=expression):
                    self.assertEqual(resolve_schedule_date.invoke({
                        'expression': expression, 'meeting_date': '2028-12-31',
                    }), f'2029-{month:02}-16')
        for weekday, particle in product(range(7), PARTICLES):
            expression = f"다음 주 {'월화수목금토일'[weekday]}요일{particle}"
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': expression, 'meeting_date': '2028-12-31',
                }), (date(2029, 1, 1) + timedelta(days=weekday)).isoformat())

    def test_particles_cannot_hide_invalid_values_or_semantic_qualifiers(self):
        conditions = ['쯤', ' 이후', ' 이전', ' 또는 모레', ' 말고 모레', '부터 모레까지', ' 취소']
        for suffix, particle in product(conditions, PARTICLES):
            with self.subTest(suffix=suffix, particle=particle):
                with self.assertRaises(ValueError):
                    resolve_schedule_date.invoke({'expression': '내일' + suffix + particle,
                                                  'meeting_date': '2028-02-28'})
                with self.assertRaises(ValueError):
                    resolve_schedule_time.invoke({'expression': '오후 4시' + suffix + particle,
                                                  'assume_business_hours': True})
        for expression, particle in product(['24시', '오전 0시', '오후 4시 60분', '23:59:60'], PARTICLES):
            with self.subTest(expression=expression, particle=particle):
                with self.assertRaises(ValueError):
                    resolve_schedule_time.invoke({'expression': expression + particle})
        for particle in PARTICLES:
            with self.assertRaises(ValueError):
                resolve_schedule_date.invoke({'expression': '2029년 2월 29일' + particle,
                                              'meeting_date': '2029-02-01'})

    def test_workflow_keeps_quoted_evidence_and_normalizes_without_model_retry(self):
        source = '납품 검수는 모레 오후 4시 17분부터 진행하기로 확정했습니다.'
        schedule = {
            'eventId': 'inspection',
            'extractedScheduleContent': '납품 검수', 'status': 'confirmed',
            'dateExpression': '모레', 'timeExpression': '오후 4시 17분부터', 'evidence': source,
        }
        content = json.dumps({'summarizedText': '검수 방법을 논의했습니다.', 'schedules': [schedule]})
        model = Mock(return_value=content)
        result = MeetingSummarizer(model, now=lambda: datetime(2028, 2, 28, 9)).summarize(source)
        self.assertEqual(result['schedules'][0]['extractedScheduleDate'], '2028-03-01T16:17:00')
        self.assertEqual(MeetingSummarizer.validate_response(content, source)['schedules'][0], schedule)
        model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
