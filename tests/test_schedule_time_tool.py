import unittest

from schedule_tools import ScheduleTimeExpressionError, resolve_schedule_time


class ScheduleTimeToolTest(unittest.TestCase):
    def test_korean_times_are_converted_to_24_hour_clock(self):
        for expression, expected in [
            ('오전 10시', '10:00:00'), ('오후 3시', '15:00:00'),
            ('오후 3시 30분', '15:30:00'), ('오전 9시 5분', '09:05:00'),
            ('오후 3시 반', '15:30:00'), (' 오전\t10시  ', '10:00:00'),
            ('오후3시30분', '15:30:00'), ('23시 59분', '23:59:00'),
            ('13시', '13:00:00'), ('0시', '00:00:00'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_time.invoke({'expression': expression}), expected)

    def test_twelve_am_and_twelve_pm_are_distinguished(self):
        for expression, expected in [
            ('오전 12시', '00:00:00'), ('오후 12시', '12:00:00'),
            ('오전 12시 반', '00:30:00'), ('오후 12시 5분', '12:05:00'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_time.invoke({'expression': expression}), expected)

    def test_colon_notation_uses_24_hour_clock_and_preserves_seconds(self):
        for expression, expected in [
            ('03:05', '03:05:00'), ('9:05', '09:05:00'), ('12:00', '12:00:00'),
            ('15:30', '15:30:00'), ('00:00:00', '00:00:00'), ('23:59:59', '23:59:59'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(resolve_schedule_time.invoke({'expression': expression}), expected)

    def test_korean_times_without_am_or_pm_are_not_guessed(self):
        for expression in ['1시', '3시', '10시 30분', '12시 반']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ScheduleTimeExpressionError, '오전 또는 오후가 없어'):
                    resolve_schedule_time.invoke({'expression': expression})

    def test_invalid_hour_minute_or_second_is_rejected(self):
        for expression in ['오전 0시', '오후 13시', '24시', '오후 3시 60분',
                           '24:00', '12:60', '12:00:60', '99:59']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ScheduleTimeExpressionError, '유효하지 않은 시각'):
                    resolve_schedule_time.invoke({'expression': expression})

    def test_qualifiers_ranges_and_references_are_not_silently_discarded(self):
        for expression in ['', ' ', '오후 3시쯤', '오후 3시 이후', '오후 3시 또는 4시',
                           '오후 3시에서 4시 사이', '15:00~16:00', '오후 3시 취소',
                           '같은 시각', '두 시간 뒤', '내일 오후 3시', '종일',
                           '오후 세 시', '3시간', '15:3', '오후 3시 반 10분']:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ScheduleTimeExpressionError, '지원하지 않는 시각 표현'):
                    resolve_schedule_time.invoke({'expression': expression})


if __name__ == '__main__':
    unittest.main()
