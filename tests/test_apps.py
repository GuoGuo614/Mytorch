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
        assert "number of epochs to train (default: 5)" in result.stdout
        assert "input batch size for training (default: 128)" in result.stdout
        assert "learning rate (default: 0.001)" in result.stdout
        assert "optimizer (default: adam)" in result.stdout
