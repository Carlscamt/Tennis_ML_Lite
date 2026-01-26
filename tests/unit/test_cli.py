import pytest
import sys
from unittest.mock import patch, MagicMock
from tennis import main

@pytest.fixture
def mock_functions():
    with patch("tennis.cmd_scrape") as mock_scrape, \
         patch("tennis.cmd_train") as mock_train, \
         patch("tennis.cmd_predict") as mock_predict, \
         patch("tennis.cmd_audit") as mock_audit, \
         patch("tennis.cmd_backtest") as mock_backtest, \
         patch("tennis.cmd_list_models") as mock_list_models, \
         patch("tennis.cmd_promote_model") as mock_promote, \
         patch("tennis.cmd_set_serving_config") as mock_serving_config, \
         patch("tennis.cmd_batch_run") as mock_batch_run, \
         patch("tennis.cmd_show_predictions") as mock_show_predictions:
        yield {
            "scrape": mock_scrape,
            "train": mock_train,
            "predict": mock_predict,
            "audit": mock_audit,
            "backtest": mock_backtest,
            "list-models": mock_list_models,
            "promote": mock_promote,
            "serving-config": mock_serving_config,
            "batch-run": mock_batch_run,
            "show-predictions": mock_show_predictions,
        }

@pytest.mark.parametrize("args,command_key", [
    (["scrape", "upcoming"], "scrape"),
    (["train"], "train"),
    (["predict"], "predict"),
    (["audit"], "audit"),
    (["backtest"], "backtest"),
    (["list-models"], "list-models"),
    (["batch-run"], "batch-run"),
    (["show-predictions"], "show-predictions"),
])
def test_cli_command_routing(mock_functions, args, command_key):
    """Verify CLI routes commands correctly to their handler functions."""
    with patch.object(sys, 'argv', ["tennis.py"] + args):
        main()
        mock_functions[command_key].assert_called_once()

def test_cli_scrape_args(mock_functions):
    """Test scrape command arguments parsing."""
    with patch.object(sys, 'argv', ["tennis.py", "scrape", "historical", "--top", "100"]):
        main()
        args = mock_functions["scrape"].call_args[0][0]
        assert args.mode == "historical"
        assert args.top == 100

def test_cli_predict_args(mock_functions):
    """Test predict command arguments parsing."""
    with patch.object(sys, 'argv', ["tennis.py", "predict", "--days", "3", "--output", "results.csv"]):
        main()
        args = mock_functions["predict"].call_args[0][0]
        assert args.days == 3
        assert args.output == "results.csv"

def test_cli_missing_command():
    """Test behavior when no command is provided."""
    with patch.object(sys, 'argv', ["tennis.py"]), pytest.raises(SystemExit):
        main()
