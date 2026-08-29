import contextlib
import io
import json
import unittest

from asd.reproduce.dspark.cli import main


class ReproductionCliTests(unittest.TestCase):
    def test_all_dry_run_is_read_only_and_lists_four_restarts(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return_code = main(
                [
                    "all",
                    "--model-path",
                    "/model",
                    "--runtime-dir",
                    "/runtime",
                    "--output-dir",
                    "/output",
                    "--dry-run",
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(result["server_restarts"], 4)
        self.assertIn("formal/asd (warmup 10 + timed 500)", result["stages"])


if __name__ == "__main__":
    unittest.main()
