"""
Pytest bootstrap.

Placed at the repo root so pytest puts the project on sys.path, letting tests
import `chatbot`, `models`, `pipeline`, ... the same way the application does.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
