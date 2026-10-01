# tests/conftest.py
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Shared test helpers that live next to the tests (e.g. exactness_referee.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
