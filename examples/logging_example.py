#!/usr/bin/env python3
"""
Example demonstrating the new pyAVS logging system.

This script shows how to:
1. Use the default logging configuration
2. Customize logging settings  
3. Use logging in your own code
4. Control logging levels dynamically
"""

import pyavs
from pyavs.utils.logging import configure_logging, get_logger, set_log_level, temporary_log_level
import tempfile
from pathlib import Path

def main():
    """Demonstrate various logging features."""
    
    print("=== pyAVS Logging System Demo ===\n")
    
    # 1. Basic usage with default configuration
    print("1. Basic usage with default INFO level:")
    logger = get_logger('demo')
    logger.debug("This DEBUG message won't be shown (level=INFO)")
    logger.info("This INFO message will be shown")
    logger.warning("This WARNING message will be shown")
    logger.error("This ERROR message will be shown")
    
    print("\n" + "-"*50 + "\n")
    
    # 2. Reconfigure logging with DEBUG level and file output
    print("2. Reconfiguring to DEBUG level with file logging:")
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "pyavs_demo.log"
        
        configure_logging(
            level='DEBUG',
            console=True,
            file_path=log_file,
            use_colors=True
        )
        
        logger = get_logger('demo.advanced')
        logger.debug("Now DEBUG messages are visible!")
        logger.info("Processing some data...")
        logger.warning("Found potential issue")
        logger.error("Error occurred during processing")
        
        # Show log file contents
        print(f"\nLog file contents ({log_file}):")
        with open(log_file, 'r') as f:
            print(f.read())
    
    print("-"*50 + "\n")
    
    # 3. Demonstrate temporary log level changes
    print("3. Temporary log level changes:")
    
    # Set back to INFO for this demo
    set_log_level('INFO')
    logger = get_logger('demo.temp')
    
    logger.debug("This DEBUG message won't be shown normally")
    
    # Temporarily enable DEBUG
    with temporary_log_level('DEBUG'):
        logger.debug("This DEBUG message is now visible inside the context!")
        logger.info("Still processing...")
    
    logger.debug("This DEBUG message is hidden again")
    logger.info("Back to normal INFO level")
    
    print("\n" + "-"*50 + "\n")
    
    # 4. Module-specific loggers
    print("4. Module-specific loggers:")
    
    # Different modules can have different loggers
    loader_logger = get_logger('dataloader.meg')
    preproc_logger = get_logger('preprocessing.trigger_tools')
    source_logger = get_logger('source.forward')
    
    loader_logger.info("Loading MEG data...")
    preproc_logger.info("Processing triggers...")
    source_logger.warning("Forward model may need updating")
    
    print("\n" + "-"*50 + "\n")
    
    # 5. Demonstrate processing logging utilities
    print("5. Processing workflow logging:")
    
    from pyavs.utils.logging import log_processing_start, log_processing_end
    import time
    
    workflow_logger = get_logger('workflow.demo')
    
    # Simulate a processing workflow
    log_processing_start(workflow_logger, "MEG preprocessing", 
                        {"subject": 1, "session": 2, "blocks": [1, 2, 3]})
    
    time.sleep(0.1)  # Simulate processing time
    
    log_processing_end(workflow_logger, "MEG preprocessing", 
                      success=True, duration=0.1, 
                      {"events_added": 150, "trials_processed": 45})
    
    # Simulate an error
    log_processing_start(workflow_logger, "source reconstruction", 
                        {"method": "beamformer"})
    
    log_processing_end(workflow_logger, "source reconstruction", 
                      success=False, duration=0.05, 
                      {"error": "Forward model not found"})
    
    print("\n" + "-"*50 + "\n")
    
    # 6. Integration with existing pyAVS functions
    print("6. Integration example:")
    print("When you call pyAVS functions, they now use proper logging:")
    print("  from pyavs.preprocessing.trigger_tools import get_avs_blocks")
    print("  blocks = get_avs_blocks(session_num=1, verbose=True)")
    print("  # This will now use logger.info() instead of print()")
    
    # Actually demonstrate if possible
    try:
        from pyavs.preprocessing.trigger_tools import get_avs_blocks
        print("\nActual output:")
        blocks = get_avs_blocks(session_num=1, verbose=True)
        print(f"Blocks for session 1: {blocks}")
    except Exception as e:
        print(f"Could not demonstrate (expected): {e}")
    
    print("\n=== Demo Complete ===")
    
    print("\nTo use logging in your own pyAVS scripts:")
    print("```python")
    print("import pyavs")
    print("from pyavs.utils.logging import get_logger, configure_logging")
    print("")
    print("# Optional: customize logging")
    print("configure_logging(level='DEBUG', file_path='my_analysis.log')")
    print("")
    print("# Get a logger for your script")
    print("logger = get_logger('my_analysis')")
    print("")
    print("# Use it throughout your code")
    print("logger.info('Starting analysis...')")
    print("logger.debug('Processing subject {}'.format(subject_id))")
    print("logger.warning('Some data missing')")
    print("```")


if __name__ == '__main__':
    main()