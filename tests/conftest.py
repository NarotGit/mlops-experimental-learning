"""
tests/conftest.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)