import base64
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from test_kernelpilot import kernelpilot as K, valid_task
import metax_reports as R
import profile_evidence as P


def kernel(start=100, duration=10, queue=0, co=1):
    return dict(
        ph="X",
        ts=start,
        dur=duration,
        name="test_kernel",
        pid=2,
        tid=1,
        args=dict(device_id=0, queue_id=queue, grid={}, block={}, co_id=co),
    )


def api(start=90, duration=5, co=1, process=123):
    return dict(
        ph="X",
        ts=start,
        dur=duration,
        name="mcLaunchKernel",
        pid=1,
        tid=1,
        args=dict(process_id=process, thread_id=123, co_id=co),
    )


class MetaXReportsTests(unittest.TestCase):
    def test_nanosecond_epoch_precision_and_correlation(self):
        epoch = 1791451023945037064
        k, a, n = R.parse_trace(
            {"traceEvents": [api(epoch, 5), kernel(epoch + 17, 10)]}, "one", "ns"
        )
        self.assertEqual(k[0]["api_end_to_kernel_start_ns"], 12)
        self.assertEqual(k[0]["duration_ns"], 10)
        self.assertEqual(k[0]["api_link_status"], "matched")

    def test_overlapping_queues_use_union_not_sum(self):
        data = {
            "traceEvents": [
                kernel(0, 10, 0, 1),
                kernel(5, 10, 0, 2),
                kernel(12, 10, 1, 3),
                kernel(30, 10, 0, 4),
            ]
        }
        report = R.summarize_timeline(*R.parse_trace(data, "one", "ns"))
        self.assertEqual(report["devices"][0]["kernel_duration_sum_ns"], 40)
        self.assertEqual(report["devices"][0]["kernel_interval_union_ns"], 32)
        self.assertEqual([g["duration_ns"] for g in report["gaps"]], [15])

    def test_ambiguous_or_missing_api_is_not_linked(self):
        k, _, _ = R.parse_trace({"traceEvents": [api(), api(), kernel()]}, "one", "ns")
        self.assertEqual(k[0]["api_link_status"], "unavailable_or_ambiguous")
        k, _, _ = R.parse_trace({"traceEvents": [kernel()]}, "two", "ns")
        self.assertNotIn("api_end_to_kernel_start_ns", k[0])

    def test_multiple_processes_are_not_inferred_from_track_pid(self):
        k, _, _ = R.parse_trace(
            {"traceEvents": [api(co=1), api(co=2, process=456), kernel()]}, "one", "ns"
        )
        self.assertNotEqual(k[0]["api_link_status"], "matched")

    def test_negative_queue_wait_is_preserved(self):
        k, _, _ = R.parse_trace(
            {"traceEvents": [api(90, 30), kernel(100, 10)]}, "one", "ns"
        )
        self.assertEqual(k[0]["api_end_to_kernel_start_ns"], -20)

    def test_missing_units_and_invalid_timestamps_fail_closed(self):
        for value, unit in [
            (1, None),
            (-1, "ns"),
            (True, "ns"),
            (float("nan"), "ns"),
            (1.8e18, "ns"),
            (0.1, "ns"),
        ]:
            with self.subTest(value=value, unit=unit), self.assertRaises(R.ReportError):
                R.timestamp(value, unit)
        self.assertEqual(R.timestamp(1.5, "us"), 1500)

    def test_empty_or_unknown_trace_fails(self):
        for data in [
            {},
            {"traceEvents": []},
            {"traceEvents": [{"ph": "X", "ts": 1, "dur": 1, "name": "unknown"}]},
        ]:
            with self.assertRaises(R.ReportError):
                R.parse_trace(data, "one", "ns")

    def test_counter_units_are_preserved_not_converted_to_latency(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "1_kernel.txt.json"
            p.write_text(
                json.dumps(
                    {
                        "Summary": [
                            dict(name="Total Cycles", value="1,234.5(Kcycles)"),
                            dict(name="efficiency", value="92.3%"),
                            dict(name="unsupported", value="N/A"),
                        ]
                    }
                )
            )
            result = R.parse_counters([p], Path(d))
            rows = result["metrics"]
            self.assertEqual((rows[0]["value"], rows[0]["unit"]), (1234.5, "Kcycles"))
            self.assertEqual(rows[1]["unit"], "%")
            self.assertEqual(rows[2]["status"], "unsupported_value")
            self.assertNotIn("kernel_timeline", result["capabilities"])

    def test_explicit_byte_suffix_and_unknown_suffixes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "1_kernel.txt.json"
            p.write_text(
                json.dumps(
                    {
                        "Memory": [
                            dict(name="read", value="32,521,792.0byte"),
                            dict(name="zero", value="0byte"),
                            dict(name="spaced", value=" 11,010,240.0 byte "),
                            dict(name="unknown", value="12widgets"),
                            dict(name="rate", value="12byte/s"),
                        ]
                    }
                )
            )
            rows = R.parse_counters([p], Path(d))["metrics"]
            self.assertEqual((rows[0]["value"], rows[0]["unit"]), (32521792.0, "byte"))
            self.assertEqual((rows[1]["value"], rows[1]["unit"]), (0.0, "byte"))
            self.assertEqual((rows[2]["value"], rows[2]["unit"]), (11010240.0, "byte"))
            self.assertEqual(rows[0]["raw_value"], "32,521,792.0byte")
            for row in rows[3:]:
                self.assertEqual(row["status"], "unsupported_value")
                self.assertIsNone(row["value"])

    def test_malformed_counter_grouping_is_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "kernel.txt.json"
            malformed = ["1,,234byte", "12,34byte", ",123byte", "123,byte"]
            entries = [dict(name="valid", value="1,234,567.5byte")]
            entries += [dict(name="invalid", value=value) for value in malformed]
            path.write_text(json.dumps({"Memory": entries}))
            rows = R.parse_counters([path], root)["metrics"]
            self.assertEqual(rows[0]["value"], 1234567.5)
            for row, raw in zip(rows[1:], malformed):
                with self.subTest(raw=raw):
                    self.assertEqual(row["status"], "unsupported_value")
                    self.assertIsNone(row["value"])
                    self.assertEqual(row["raw_value"], raw)

    def test_nonfinite_byte_suffix_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "1_kernel.txt.json"
            p.write_text(json.dumps({"Memory": [dict(name="read", value="1e309byte")]}))
            with self.assertRaisesRegex(R.ReportError, "nonfinite"):
                R.parse_counters([p], Path(d))

    def test_scope_excludes_warmup_and_requires_completed_correlated_work(self):
        start = api(100, 2, 90)
        start["name"] = "mcProfilerStart"
        stop = api(200, 2, 91)
        stop["name"] = "mcProfilerStop"
        events = [
            api(10, 2, 1),
            kernel(20, 10, co=1),
            start,
            api(120, 2, 2),
            kernel(130, 10, co=2),
            stop,
        ]
        kernels, apis, _ = R.parse_trace(dict(traceEvents=events), "one", "ns")
        selected, _, ranges = R.select_range(
            kernels, apis, dict(start_api="mcProfilerStart", stop_api="mcProfilerStop")
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["start_ns"], 130)
        self.assertEqual(ranges[0]["kernel_count"], 1)
        kernels[1]["end_ns"] = 210
        with self.assertRaisesRegex(R.ReportError, "boundary-crossing"):
            R.select_range(
                kernels,
                apis,
                dict(start_api="mcProfilerStart", stop_api="mcProfilerStop"),
            )

    def test_missing_scope_marker_or_kernel_identity_fails(self):
        kernels, apis, _ = R.parse_trace(
            dict(traceEvents=[api(), kernel()]), "one", "ns"
        )
        with self.assertRaises(R.ReportError):
            R.select_range(kernels, apis, dict(start_api="start", stop_api="stop"))
        event = kernel()
        del event["args"]["queue_id"]
        with self.assertRaisesRegex(R.ReportError, "identity"):
            R.parse_trace(dict(traceEvents=[event]), "one", "ns")

    def test_no_export_is_failure_even_if_tool_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            for tool in ("mctracer", "mcprofiler"):
                with self.assertRaises(R.ReportError):
                    R.normalize(Path(d), tool, "ns")

    def test_archive_traversal_and_hash_mismatch_rejected(self):
        for name, hash_ in [("../escape", "x"), ("safe", "wrong")]:
            with tempfile.TemporaryDirectory() as d:
                b = io.BytesIO()
                with tarfile.open(fileobj=b, mode="w:gz") as t:
                    item = tarfile.TarInfo(name)
                    item.size = 1
                    t.addfile(item, io.BytesIO(b"x"))
                envelope = {
                    "archive_base64": base64.b64encode(b.getvalue()).decode(),
                    "files": {name: hash_},
                }
                with self.assertRaises(K.evidence.EvidenceError):
                    P.unpack(envelope, Path(d), 10000)


FAKE = """#!/usr/bin/env python3
from pathlib import Path
import json,os,subprocess,sys,uuid
if '--help' in sys.argv or 'version' in sys.argv:
 print('fake profiler 1.0');raise SystemExit(0)
tracer='tracer' in Path(__file__).name
if tracer:
 i=sys.argv.index('--name')+2;subprocess.run(sys.argv[i:])
 if not os.environ.get('KDA_FAKE_EMPTY'):
  p=Path('trace');p.mkdir()
  events=[dict(ph='X',ts=100,dur=10,name='fake_kernel',args=dict(device_id=0,queue_id=0,grid={},block={},co_id=1)),dict(ph='X',ts=90,dur=5,name='launch',args=dict(process_id=1,thread_id=1,co_id=1))]
  (p/'trace.json').write_text(json.dumps(dict(traceEvents=events)))
else:
 subprocess.run(['bash','-c',sys.argv[sys.argv.index('--cmdline')+1]])
 p=Path(__file__).parent/('output'+uuid.uuid4().hex);p.mkdir()
 (p/'1_kernel.txt.json').write_text(json.dumps({'Summary':[dict(name='Total Cycles',value='12.5(Kcycles)')]}))
 print('perf done, please check report file '+str(p))
"""


class MetaXProfileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        for name, value in [
            ("kernel.py", "pass\n"),
            ("workload.py", 'print("correct")\n'),
            (
                "guard.py",
                'import json; print(json.dumps(dict(idle=True,environment_id="fixture")))\n',
            ),
        ]:
            (self.ws / name).write_text(value)
        for tool in ("mcprofiler", "mctracer"):
            p = self.ws / ("fake-" + tool)
            p.write_text(FAKE)
            p.chmod(0o755)
        self.task = valid_task()
        self.task["target"] = dict(backend="maca", device="test GPU")
        self.task["environment"] = {
            "setup": f'export KDA_MCPROFILER_BIN="{self.ws}/fake-mcprofiler"\nexport KDA_MCTRACER_BIN="{self.ws}/fake-mctracer"'
        }
        self.task["constraints"]["forbidden_paths"] = ["guard.py", "workload.py"]
        self.task["profile"] = {
            "tools": {
                tool: dict(
                    adapter="metax",
                    workload_command="python3 workload.py",
                    guard_command="python3 guard.py",
                    **({"timestamp_unit": "ns"} if tool == "mctracer" else {}),
                )
                for tool in ("mcprofiler", "mctracer")
            }
        }

    def capture(self, tool="mctracer"):
        K.discover_analysis_tools(self.ws, self.task)
        return K.run_contract_command(
            self.ws, self.task, "profile", tool=tool, capture=True
        )

    def records(self):
        return [
            json.loads(p.read_text()) for p in (self.ws / P.ROOT).glob("*/record.json")
        ]

    def test_maca_discovers_both_tools_as_unverified(self):
        result = K.discover_analysis_tools(self.ws, self.task)
        self.assertEqual(result["backend_profile"], "maca")
        self.assertEqual(
            {t["id"] for t in result["tools"] if t["available"]},
            {"mcprofiler", "mctracer"},
        )
        self.assertTrue(
            all(
                t["capability_status"] == "declared_unverified" for t in result["tools"]
            )
        )

    def test_local_capture_export_parse_and_hash_binding(self):
        for tool in ("mctracer", "mcprofiler"):
            result = self.capture(tool)
            self.assertEqual(result.returncode, 0)
        records = self.records()
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["status"], "pass")
            self.assertEqual(record["before"], record["after"])
            self.assertEqual(record["purpose"], "diagnostic_only_not_promotion")
            self.assertTrue(record["validated_capabilities"])
            K.evidence.load_record(self.ws / P.ROOT / record["id"])
        self.assertFalse((self.ws / ".kernelpilot/checkpoints/best").exists())

    def test_missing_discovery_blocks_execution(self):
        with self.assertRaisesRegex(K.ContractError, "discover-tools"):
            K.run_contract_command(
                self.ws, self.task, "profile", tool="mctracer", capture=True
            )
        self.assertFalse((self.ws / P.ROOT).exists())

    def test_zero_exit_missing_trace_preserves_error(self):
        self.task["environment"]["setup"] += "\nexport KDA_FAKE_EMPTY=1"
        with self.assertRaisesRegex(K.evidence.EvidenceError, "no trace JSON"):
            self.capture()
        record = self.records()[0]
        self.assertEqual(record["status"], "error")
        self.assertIn("raw/capture.json", record["artifacts"])

    def test_profiled_target_failure_is_not_hidden_by_tool_zero_exit(self):
        (self.ws / "workload.py").write_text("raise SystemExit(7)")
        with self.assertRaisesRegex(K.evidence.EvidenceError, "workload failed"):
            self.capture()
        self.assertEqual(self.records()[0]["status"], "error")

    def test_busy_guard_stops_capture(self):
        (self.ws / "guard.py").write_text(
            'import json;print(json.dumps(dict(idle=False,environment_id="fixture")))'
        )
        with self.assertRaisesRegex(K.evidence.EvidenceError, "busy"):
            self.capture()
        self.assertFalse(any("capture" in record for record in self.records()))

    def test_source_change_rejected_with_raw_evidence(self):
        (self.ws / "workload.py").write_text(
            'from pathlib import Path;Path("kernel.py").write_text("changed")'
        )
        with self.assertRaisesRegex(K.evidence.EvidenceError, "source/protected"):
            self.capture()
        self.assertIn("raw/capture.json", self.records()[0]["artifacts"])

    def test_stale_discovery_rejected(self):
        K.discover_analysis_tools(self.ws, self.task)
        self.task["environment"]["setup"] += "\nexport NEW_ENV=1"
        with self.assertRaisesRegex(K.ContractError, "changed task environment"):
            K.run_contract_command(
                self.ws, self.task, "profile", tool="mctracer", capture=True
            )

    def test_post_guard_busy_preserves_collected_raw_trace(self):
        (self.ws / "workload.py").write_text(
            'from pathlib import Path;Path("busy").write_text("1")'
        )
        (self.ws / "guard.py").write_text(
            'import json;from pathlib import Path;print(json.dumps(dict(idle=not Path("busy").exists(),environment_id="fixture")))'
        )
        with self.assertRaisesRegex(K.evidence.EvidenceError, "busy"):
            self.capture()
        self.assertIn("raw/trace/trace.json", self.records()[0]["artifacts"])

    def test_profile_report_verifies_artifacts_and_rejects_tampering(self):
        result = self.capture()
        run = json.loads(result.stdout)["profile_id"]
        summary = P.report(self.ws, run)
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["kernel_count"], 1)
        self.assertEqual(summary["correlation_coverage"], dict(matched=1, total=1))
        (self.ws / P.ROOT / run / "analysis/kernels.csv").write_text("tampered")
        with self.assertRaisesRegex(K.evidence.EvidenceError, "hash mismatch"):
            P.report(self.ws, run)

    def test_invalid_config_rejected(self):
        for extra in [
            dict(counts=0),
            dict(timestamp_unit=None),
            dict(timestamp_unit="us"),
            dict(metrics=["--remote_pwd"]),
            dict(typo=True),
        ]:
            config = dict(
                adapter="metax",
                workload_command="true",
                guard_command="true",
                timestamp_unit="ns",
                **{},
            )
            config.update(extra)
            with self.assertRaises(K.evidence.EvidenceError):
                P.validate_config(config, "mctracer")

    def test_missing_requested_counter_retains_raw_and_rejects_success(self):
        self.task["profile"]["tools"]["mcprofiler"]["metrics"] = [
            "Total Cycles",
            "Missing Metric",
        ]
        with self.assertRaisesRegex(K.evidence.EvidenceError, "metrics are missing"):
            self.capture("mcprofiler")
        record = self.records()[0]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["metric_coverage"]["unavailable"], ["Missing Metric"])
        self.assertIn("analysis/metrics.csv", record["artifacts"])
        self.assertIn("raw/capture.json", record["artifacts"])

    def test_counter_missing_from_one_scope_is_not_hidden_by_aggregate(self):
        binary = self.ws / "fake-mcprofiler"
        script = binary.read_text().replace(
            "print('perf done, please check report file '+str(p))",
            "(p/'2_kernel.txt.json').write_text(json.dumps({'Summary':[dict(name='Other',value='3')]}))\n print('perf done, please check report file '+str(p))",
        )
        binary.write_text(script)
        with self.assertRaisesRegex(K.evidence.EvidenceError, "metrics are missing"):
            self.capture("mcprofiler")
        record = self.records()[0]
        self.assertEqual(record["metric_coverage"]["unavailable"], [])
        self.assertEqual(
            list(record["metric_coverage"]["missing_by_scope"].values()),
            [["Total Cycles"]],
        )
