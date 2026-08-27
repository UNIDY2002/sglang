"""Unit tests for kill_process_tree's reap wait — no server, no model loading."""

import subprocess
import sys
import unittest

import psutil

from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A launched server is a parent owning GPU-holding children; the parent here
# reports its child's pid so the test can watch the child independently.
PARENT_SCRIPT = (
    "import subprocess, sys, time; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']); "
    "print(child.pid, flush=True); "
    "time.sleep(300)"
)


class TestKillProcessTreeWait(CustomTestCase):
    def test_returns_only_after_the_tree_is_released(self):
        """Teardown must not return while the tree still holds its resources.

        A fire-and-forget kill left a SIGKILLed server's GPU memory and ports
        held past the call, so the next fixture booted onto a GPU that was
        still being reclaimed.
        """
        process = subprocess.Popen(
            [sys.executable, "-c", PARENT_SCRIPT], stdout=subprocess.PIPE, text=True
        )
        self.addCleanup(process.stdout.close)
        self.addCleanup(kill_process_tree, process.pid, wait_timeout=None)

        reported_pid = process.stdout.readline()
        self.assertTrue(reported_pid, "parent exited before reporting its child pid")
        child = psutil.Process(int(reported_pid))
        self.addCleanup(kill_process_tree, child.pid, wait_timeout=None)
        parent = psutil.Process(process.pid)

        kill_process_tree(process.pid)

        for proc, label in ((parent, "parent"), (child, "child")):
            # A zombie has already had its resources freed by the kernel.
            try:
                holding = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, OSError):
                holding = False
            self.assertFalse(holding, f"{label} {proc.pid} still holds resources")


if __name__ == "__main__":
    unittest.main()
