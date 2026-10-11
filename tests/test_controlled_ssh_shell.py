import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from controlled_ssh import ControlledSSH, SSHAdapterError


class LocalShellTransport(ControlledSSH):
    def _argv(self, script, shell=None):
        return ["bash", "-lc", script]


class DirectoryOperandTest(unittest.TestCase):
    def test_actual_push_script_preserves_hash_guard_and_replaces_source(self):
        with tempfile.TemporaryDirectory(prefix="kda directory ") as directory:
            root = Path(directory)
            (root / "src").mkdir()
            target = root / "src/kernel.py"
            target.write_bytes(b"original")
            candidate = root / "candidate.py"
            candidate.write_bytes(b"candidate")
            task = {
                "execution": {
                    "transport": "ssh",
                    "host": "test-host",
                    "workspace": str(root),
                },
                "constraints": {"allowed_paths": ["src/kernel.py"]},
            }
            adapter = LocalShellTransport(task)
            old_hash = hashlib.sha256(b"original").hexdigest()
            new_hash = adapter.push("src/kernel.py", candidate, old_hash)
            self.assertEqual(target.read_bytes(), b"candidate")
            self.assertEqual(new_hash, hashlib.sha256(b"candidate").hexdigest())
            with self.assertRaises(SSHAdapterError):
                adapter.push("src/kernel.py", candidate, old_hash)
            self.assertEqual(target.read_bytes(), b"candidate")
