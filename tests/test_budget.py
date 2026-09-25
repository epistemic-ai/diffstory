import unittest

from diffstory.budget import BudgetLimits, INPUT_FRAMING_MARGIN, RunBudget


class BudgetTests(unittest.TestCase):
    def limits(self, **updates):
        values = {
            "context_tokens": 12_000,
            "request_input_tokens": 8_000,
            "request_output_tokens": 3_000,
            "total_input_tokens": 12_000,
            "total_output_tokens": 6_000,
            "calls": 4,
            "seconds": 10,
        }
        values.update(updates)
        return BudgetLimits(**values)

    def test_serialized_bytes_and_framing_are_reserved(self):
        budget = RunBudget(self.limits())
        body = b'{"model":"m","input":"source","text":{"format":{}}}'
        reservation = budget.authorize(body, 1000)
        self.assertEqual(reservation.input_tokens, len(body) + INPUT_FRAMING_MARGIN)
        self.assertEqual(budget.usage()["input_tokens"], reservation.input_tokens)
        self.assertEqual(budget.usage()["output_tokens"], 1000)

    def test_context_and_per_request_limits_fail_before_reservation(self):
        with self.assertRaisesRegex(ValueError, "exceed the model context"):
            self.limits(context_tokens=5000, request_input_tokens=2500, request_output_tokens=3000)
        per_request = RunBudget(self.limits(request_input_tokens=3000))
        with self.assertRaisesRegex(ValueError, "per-request budget"):
            per_request.authorize(b"x" * 3000, 1000)
        self.assertEqual(per_request.usage()["calls"], 0)

    def test_output_reservation_is_checked_before_dispatch(self):
        budget = RunBudget(self.limits(total_output_tokens=1000))
        with self.assertRaisesRegex(ValueError, "output-token budget"):
            budget.authorize(b"small", 1001)
        self.assertEqual(budget.usage()["calls"], 0)

    def test_actual_usage_reconciles_reservation(self):
        budget = RunBudget(self.limits())
        reservation = budget.authorize(b"body", 1000)
        budget.settle(reservation, {"input_tokens": 200, "output_tokens": 150})
        self.assertEqual(budget.usage()["input_tokens"], 200)
        self.assertEqual(budget.usage()["output_tokens"], 150)

    def test_usage_above_reservation_fails_closed(self):
        budget = RunBudget(self.limits())
        reservation = budget.authorize(b"body", 1000)
        with self.assertRaisesRegex(ValueError, "exceeded the authorized request reservation"):
            budget.settle(reservation, {"input_tokens": reservation.input_tokens, "output_tokens": 1001})
        self.assertEqual(budget.usage()["output_tokens"], 1001)

    def test_unknown_usage_keeps_full_reservation(self):
        budget = RunBudget(self.limits())
        reservation = budget.authorize(b"body", 1000)
        budget.settle(reservation, None)
        self.assertEqual(budget.usage()["input_tokens"], reservation.input_tokens)
        self.assertEqual(budget.usage()["output_tokens"], 1000)
        self.assertEqual(budget.usage()["calls"], 1)

    def test_retry_requires_and_consumes_a_second_reservation(self):
        budget = RunBudget(self.limits(calls=1))
        first = budget.authorize(b"body", 1000)
        budget.settle(first, {"input_tokens": 200, "output_tokens": 100})
        with self.assertRaisesRegex(ValueError, "call budget exhausted"):
            budget.authorize(b"retry", 1000)
        self.assertEqual(budget.usage()["calls"], 1)

    def test_call_budget_rejects_before_an_extra_call(self):
        budget = RunBudget(self.limits(calls=1))
        first = budget.authorize(b"body", 1000)
        budget.settle(first, None)
        with self.assertRaisesRegex(ValueError, "call budget exhausted"):
            budget.authorize(b"body", 1000)
        self.assertEqual(budget.usage()["calls"], 1)

    def test_deadline_is_checked_before_reservation(self):
        now = [0.0]
        budget = RunBudget(self.limits(seconds=2), clock=lambda: now[0])
        reservation = budget.authorize(b"first", 1000)
        self.assertEqual(reservation.timeout_seconds, 2)
        budget.settle(reservation, None)
        now[0] = 2.1
        with self.assertRaisesRegex(ValueError, "time budget exhausted"):
            budget.authorize(b"second", 1000)
        self.assertEqual(budget.usage()["calls"], 1)

    def test_cancel_before_dispatch_releases_budget(self):
        budget = RunBudget(self.limits(calls=1))
        reservation = budget.authorize(b"body", 1000)
        budget.cancel_before_dispatch(reservation)
        self.assertEqual(budget.usage()["calls"], 0)
        self.assertEqual(budget.usage()["input_tokens"], 0)
        self.assertEqual(budget.usage()["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
