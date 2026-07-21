import argparse
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app import _loopback_host


class ApplicationEntrypointTests(unittest.TestCase):
    def test_localhost_bindings_are_allowed(self) -> None:
        self.assertEqual("127.0.0.1", _loopback_host("127.0.0.1"))
        self.assertEqual("::1", _loopback_host("::1"))
        self.assertEqual("localhost", _loopback_host("localhost"))

    def test_network_binding_is_rejected_without_authentication(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _loopback_host("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
