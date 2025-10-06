#!/usr/bin/env python3
"""
Test script to verify the CLI functionality
"""

import sys
import os

# Add the current directory to the path so we can import yahtzotron
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from yahtzotron.cli import cli
    print("✅ CLI imported successfully")
    
    # Test that the CLI group name is correct
    if cli.name == "yum-tron":
        print("✅ CLI group name is correct: 'yum-tron'")
    else:
        print(f"❌ CLI group name is incorrect: '{cli.name}' (expected 'yum-tron')")
    
    # Test that the help text is correct by checking the docstring
    if "Yum-tron" in cli.help:
        print("✅ CLI help text contains 'Yum-tron'")
    else:
        print("❌ CLI help text does not contain 'Yum-tron'")
    
    # Test that the default ruleset is yum
    from yahtzotron.cli import train
    if hasattr(train, 'params'):
        for param in train.params:
            if param.name == 'ruleset' and param.default == 'yum':
                print("✅ Default ruleset is 'yum'")
                break
        else:
            print("❌ Default ruleset is not 'yum'")
    
    # Test that the origin command description is correct
    from yahtzotron.cli import origin
    if "Yum-tron" in origin.help:
        print("✅ Origin command help contains 'Yum-tron'")
    else:
        print("❌ Origin command help does not contain 'Yum-tron'")
    
    # Test that the play command description is correct
    from yahtzotron.cli import play
    if "Yum-tron" in play.help:
        print("✅ Play command help contains 'Yum-tron'")
    else:
        print("❌ Play command help does not contain 'Yum-tron'")
    
    print("\nCLI test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing CLI: {e}")
    import traceback
    traceback.print_exc()
