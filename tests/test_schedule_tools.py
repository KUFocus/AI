import unittest

from schedule_tools import resolve_schedule_date


class ScheduleDateToolTest(unittest.TestCase):
    def test_resolves_day_offsets_including_leap_day_and_year_change(self):
        for expression, reference, expected in [
            ('오늘', '2026-09-25', '2026-09-25'),
            ('내일', '2026-09-25', '2026-09-26'),
            ('모레', '2026-09-25', '2026-09-27'),
            ('내일', '2028-02-28', '2028-02-29'),
            ('내일', '2026-02-28', '2026-03-01'),
            ('모레', '2026-12-31', '2027-01-02'),
        ]:
            with self.subTest(expression=expression, reference=reference):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': expression, 'meeting_date': reference,
                }), expected)

    def test_next_week_starts_on_monday(self):
        for weekday, expected in [
            ('월', '2026-09-28'), ('화', '2026-09-29'), ('수', '2026-09-30'),
            ('목', '2026-10-01'), ('금', '2026-10-02'), ('토', '2026-10-03'), ('일', '2026-10-04'),
        ]:
            with self.subTest(weekday=weekday):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': f'다음 주 {weekday}요일', 'meeting_date': '2026-09-25',
                }), expected)
        for reference, expected in [
            ('2026-09-27', '2026-09-28'),
            ('2026-09-28', '2026-10-05'),
            ('2026-12-31', '2027-01-04'),
        ]:
            with self.subTest(reference=reference):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': ' 다음주월요일 ', 'meeting_date': reference,
                }), expected)

    def test_unsupported_or_partial_expressions_are_rejected(self):
        for expression in ['', '월요일', '이번 주 월요일', '다음 주', '내일이나 모레',
                           '내일 오전 10시 또는 모레', '내일 오전 10시 이후', '내일 말고 모레',
                           '다음 주 월요일 오전 10시 취소', '9월 28일']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, '지원하지 않는 날짜 표현'):
                    resolve_schedule_date.invoke({'expression': expression, 'meeting_date': '2026-09-25'})

    def test_time_suffix_does_not_bypass_relative_date_calculation(self):
        for expression, expected in [
            ('다음 주 월요일 오전 10시', '2026-09-28'),
            ('다음 주 화요일 오후 3시 30분', '2026-09-29'),
            ('다음 주 수요일 오전 11시 반', '2026-09-30'),
            ('오늘 0시', '2026-09-25'),
            ('내일 23시 59분', '2026-09-26'),
            ('모레 오후 12시', '2026-09-27'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': expression, 'meeting_date': '2026-09-25',
                }), expected)

    def test_explicit_dates_are_resolved_without_changing_year(self):
        for expression, expected in [
            ('2026년 9월 28일', '2026-09-28'),
            ('2026년 9월 28일 오전 10시', '2026-09-28'),
            ('2028년 2월 29일 오후 3시', '2028-02-29'),
            ('2028-02-29', '2028-02-29'),
            ('2026-09-28 오후 3시', '2026-09-28'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_date.invoke({
                    'expression': expression, 'meeting_date': '2026-09-25',
                }), expected)

    def test_invalid_date_or_time_in_original_expression_is_rejected(self):
        for expression in ['2026년 2월 29일', '2026-02-30', '0000년 1월 1일',
                           '내일 오전 13시', '내일 오후 0시', '내일 24시', '내일 10시 60분']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, '유효하지 않은'):
                    resolve_schedule_date.invoke({'expression': expression, 'meeting_date': '2026-09-25'})

    def test_invalid_reference_dates_are_rejected(self):
        for reference in ['2026-02-30', '20260925', '2026/09/25', '2026-09-25T10:00:00', '']:
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ValueError, '회의 기준일은 YYYY-MM-DD 형식'):
                    resolve_schedule_date.invoke({'expression': '내일', 'meeting_date': reference})

    def test_out_of_range_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '지원 범위를 벗어났습니다'):
            resolve_schedule_date.invoke({'expression': '내일', 'meeting_date': '9999-12-31'})


if __name__ == '__main__':
    unittest.main()
