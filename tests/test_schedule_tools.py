import unittest

from schedule_tools import resolve_relative_date


class RelativeDateToolTest(unittest.TestCase):
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
                self.assertEqual(resolve_relative_date.invoke({
                    'expression': expression, 'meeting_date': reference,
                }), expected)

    def test_next_week_starts_on_monday(self):
        for weekday, expected in [
            ('월', '2026-09-28'), ('화', '2026-09-29'), ('수', '2026-09-30'),
            ('목', '2026-10-01'), ('금', '2026-10-02'), ('토', '2026-10-03'), ('일', '2026-10-04'),
        ]:
            with self.subTest(weekday=weekday):
                self.assertEqual(resolve_relative_date.invoke({
                    'expression': f'다음 주 {weekday}요일', 'meeting_date': '2026-09-25',
                }), expected)
        for reference, expected in [
            ('2026-09-27', '2026-09-28'),
            ('2026-09-28', '2026-10-05'),
            ('2026-12-31', '2027-01-04'),
        ]:
            with self.subTest(reference=reference):
                self.assertEqual(resolve_relative_date.invoke({
                    'expression': ' 다음주월요일 ', 'meeting_date': reference,
                }), expected)

    def test_unsupported_or_partial_expressions_are_rejected(self):
        for expression in ['', '월요일', '이번 주 월요일', '다음 주', '내일이나 모레', '다음 주 월요일 오전 10시']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, '지원하지 않는 상대 날짜 표현'):
                    resolve_relative_date.invoke({'expression': expression, 'meeting_date': '2026-09-25'})

    def test_invalid_reference_dates_are_rejected(self):
        for reference in ['2026-02-30', '20260925', '2026/09/25', '2026-09-25T10:00:00', '']:
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ValueError, '회의 기준일은 YYYY-MM-DD 형식'):
                    resolve_relative_date.invoke({'expression': '내일', 'meeting_date': reference})

    def test_out_of_range_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '지원 범위를 벗어났습니다'):
            resolve_relative_date.invoke({'expression': '내일', 'meeting_date': '9999-12-31'})


if __name__ == '__main__':
    unittest.main()
