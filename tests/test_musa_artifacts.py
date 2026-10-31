import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "inspect_musa_artifacts",
    Path(__file__).resolve().parents[1] / "scripts/inspect_musa_artifacts.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MusaArtifactTests(unittest.TestCase):
    def test_false_metadata_does_not_hide_real_sqmma_call(self):
        ir = "%x = tail call <8 x i32> @llvm.musa.sqmma.bfmma.m32n32k64.mma(i32 %a)"
        result = module.inspect_artifacts({"uses_sqmma": False}, ir)
        self.assertTrue(result["metadata_ir_disagree"])
        self.assertEqual(len(result["observed_sqmma_intrinsic_calls"]), 1)

    def test_unused_declaration_is_not_execution_evidence(self):
        ir = "declare <8 x i32> @llvm.musa.sqmma.bfmma.m32n32k64.mma(i32)"
        result = module.inspect_artifacts({"uses_sqmma": False}, ir)
        self.assertFalse(result["metadata_ir_disagree"])
        self.assertEqual(result["observed_sqmma_intrinsic_calls"], [])

    def test_missing_flag_is_unknown_not_false(self):
        ir = "%x = call <8 x i32> @llvm.musa.sqmma.bfmma.m32n32k64.mma(i32 %a)"
        result = module.inspect_artifacts({}, ir)
        self.assertIsNone(result["metadata_uses_sqmma"])
        self.assertFalse(result["metadata_ir_disagree"])
