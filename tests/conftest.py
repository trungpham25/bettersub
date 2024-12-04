import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add auto_avsr directory to Python path
auto_avsr_dir = os.path.join(project_root, "auto_avsr")
sys.path.insert(0, auto_avsr_dir)
