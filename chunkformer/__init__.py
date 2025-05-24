# Make the chunkformer directory a proper Python package
import os
import sys

# Add parent directory of 'model' module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) 