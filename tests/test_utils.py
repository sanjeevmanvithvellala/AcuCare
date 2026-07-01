import os
from app.utils.config_loader import load_config
from app.utils.logger import setup_logger

def test_load_config():
    """Test that the configuration loads properly and resolves relative paths."""
    config = load_config()
    assert "project" in config
    assert "paths" in config
    assert "models" in config
    
    # Check resolved paths are absolute
    for path_key, path_val in config["paths"].items():
        assert os.path.isabs(path_val)

def test_setup_logger(tmp_path):
    """Test logger setup and rotating file creation."""
    log_file = os.path.join(tmp_path, "test.log")
    test_logger = setup_logger("test_logger", log_file)
    
    # Log a message
    msg = "Test log message"
    test_logger.info(msg)
    
    # Verify log file exists and contains the message
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert msg in content
