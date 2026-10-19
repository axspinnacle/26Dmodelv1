"""
Scoring utilities for Stage 08.

Provides functions for loading, preparing, and scoring new data using 
trained models. Supports two GLM prediction sources:
- "lookup": Join with control file using join_key (for historical data with vin_date)
- "column": GLM predictions already included in input data (for new data)

All functions include validation, clear error messages, and logging.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
