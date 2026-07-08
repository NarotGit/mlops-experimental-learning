"""
tests/conftest.py
"""
import os
import sys

# Make sure project root is on the path so all imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)