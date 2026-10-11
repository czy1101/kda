import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from test_kernelpilot import kernelpilot, valid_task


class EvaluationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        (self.workspace / "runs").mkdir()
        self.task = valid_task()
        self.task["benchmark"]["shapes"] = ["forward/fp16/a", "backward/bf16/b"]
        self.task["goal"].update(aggregation="geomean_ratio", max_regression=0.02)
        self.rows = [
            dict(
                candidate="c1",
                shape=shape,
                metric_value=8,
                baseline_value=10,
                unit="ms",
                status="pass",
            )
            for shape in self.task["benchmark"]["shapes"]
        ]

    def write(self):
        with (self.workspace / "runs/benchmark.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def evaluate(self):
        self.write()
        return kernelpilot.evaluate_goal(self.workspace, self.task, "c1")

    def test_missing_ledger_fails_closed(self):
        result, errors = kernelpilot.evaluate_goal(self.workspace, self.task, "c1")
        self.assertIsNone(result)
        self.assertIn("missing benchmark ledger", errors[0])

    def test_duplicate_summary_cannot_hide_failure(self):
        self.rows.insert(0, dict(self.rows[0], status="fail"))
        result, errors = self.evaluate()
        self.assertIsNone(result)
        self.assertIn("duplicate", errors[0])

    def test_duplicates_without_required_shapes_are_also_rejected(self):
        self.task["benchmark"].pop("shapes")
        self.rows.append(self.rows[0].copy())
        self.assertIn("duplicate", self.evaluate()[1][0])

    def test_nonfinite_nonpositive_and_missing_values(self):
        for field in ("metric_value", "baseline_value"):
            for value in ("nan", "inf", "-inf", 0, -1, ""):
                with self.subTest(field=field, value=value):
                    original = self.rows[0][field]
                    self.rows[0][field] = value
                    self.assertIsNone(self.evaluate()[0])
                    self.rows[0][field] = original

    def test_mixed_or_empty_units_rejected(self):
        for unit in ("us", "", " "):
            self.rows[1]["unit"] = unit
            self.assertIn("unit", self.evaluate()[1][0])

    def test_required_workload_cannot_be_missing(self):
        self.rows.pop()
        self.assertIn("required shapes", self.evaluate()[1][0])

    def test_duplicate_contract_shapes_rejected(self):
        self.task["benchmark"]["shapes"].append(self.task["benchmark"]["shapes"][0])
        self.assertIn("duplicate", self.evaluate()[1][0])

    def test_reordered_rows_keep_workload_identity(self):
        self.rows[1]["metric_value"] = 9
        self.rows.reverse()
        result, errors = self.evaluate()
        self.assertEqual(errors, [])
        self.assertEqual(
            [r["shape"] for r in result["per_shape"]], self.task["benchmark"]["shapes"]
        )
        self.assertEqual(result["per_shape_ratios"], [0.8, 0.9])
        self.assertTrue(result["target_reached"])

    def test_single_regression_blocks_geomean_winner(self):
        self.rows[0]["metric_value"] = 1
        self.rows[1]["metric_value"] = 11
        result, errors = self.evaluate()
        self.assertEqual(errors, [])
        self.assertLess(result["aggregate_ratio"], 0.9)
        self.assertFalse(result["target_reached"])
        self.assertFalse(result["per_shape"][1]["regression_ok"])

    def test_throughput_maximize_uses_same_generic_evaluator(self):
        self.task["goal"].update(
            metric="throughput", direction="maximize", relative_to_baseline=1.1
        )
        for row in self.rows:
            row.update(metric_value=12, unit="items/s")
        result, errors = self.evaluate()
        self.assertEqual(errors, [])
        self.assertTrue(result["target_reached"])
        self.rows[1]["metric_value"] = 9
        self.assertFalse(self.evaluate()[0]["target_reached"])

    def test_invalid_goal_and_numeric_overflow_rejected(self):
        self.task["goal"]["relative_to_baseline"] = float("nan")
        self.assertIsNone(self.evaluate()[0])
        self.task["goal"]["relative_to_baseline"] = 0.9
        self.rows[0].update(metric_value=1e308, baseline_value=1e-308)
        self.assertIsNone(self.evaluate()[0])

    def test_cli_exit_codes_and_json(self):
        task_file = self.workspace / "task.yaml"
        task_file.write_text(yaml.safe_dump(self.task))
        args = [
            "evaluate",
            "--workspace",
            str(self.workspace),
            "--task",
            str(task_file),
            "--candidate",
            "c1",
        ]
        for value, expected in ((8, 0), (10, 1), ("nan", 2)):
            self.rows[0]["metric_value"] = value
            if value == 10:
                self.rows[1]["metric_value"] = 10
            self.write()
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = kernelpilot.main(args)
            self.assertEqual(code, expected)
            if code < 2:
                self.assertEqual(
                    json.loads(stdout.getvalue())["target_reached"], code == 0
                )
            else:
                self.assertIn("invalid benchmark evidence", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
