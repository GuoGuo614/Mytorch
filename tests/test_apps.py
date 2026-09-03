import subprocess
import sys


def test_mnist_entrypoints_show_help_without_loading_data():
    for module in ("apps.mlp_mnist", "apps.lenet5_mnist"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout.lower()
