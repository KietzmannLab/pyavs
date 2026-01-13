"""
Message parsing functions for EyeLink calibration and drift correction data.

This module provides functions to parse and extract calibration validation
metrics and drift correction offsets from EyeLink messages CSV files.
"""

import pandas as pd
import numpy as np
import re
import ast
from typing import Any, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def flatten_nested_lists(data: Any, max_depth: int = 10) -> List:
    """
    Recursively flatten deeply nested lists.

    EyeLink CAL and VALIDATION messages may be stored as deeply nested lists
    due to repeated parsing or accumulation. This function flattens them to
    extract individual tokens.

    Parameters
    ----------
    data : Any
        Data to flatten (can be list, tuple, or other)
    max_depth : int, default=10
        Maximum recursion depth to prevent infinite loops

    Returns
    -------
    List
        Flattened list of tokens

    Examples
    --------
    >>> flatten_nested_lists([['a', ['b', 'c']], 'd'])
    ['a', 'b', 'c', 'd']
    """
    if max_depth == 0:
        return [data]

    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            result.extend(flatten_nested_lists(item, max_depth - 1))
        return result
    else:
        return [data]


def parse_drift_correction_message(drift_msg: str) -> Optional[Dict]:
    """
    Parse DRIFTCORRECT message string using regex.

    Extracts position offsets in both degrees and pixels from EyeLink
    drift correction messages.

    Parameters
    ----------
    drift_msg : str
        DRIFTCORRECT message string
        Example: "DRIFTCORRECT L LEFT  at 512,384  OFFSET 0.13 deg.  4.0,0.7 pix."

    Returns
    -------
    dict or None
        Dictionary with parsed values:
        - check_x : int - X coordinate of check position
        - check_y : int - Y coordinate of check position
        - offset_deg : float - Total offset in degrees
        - offset_x_pix : float - X offset in pixels
        - offset_y_pix : float - Y offset in pixels
        - offset_total_deg : float - Same as offset_deg (for consistency)
        - offset_total_pix : float - Euclidean distance in pixels
        Returns None if parsing fails

    Examples
    --------
    >>> msg = "DRIFTCORRECT L LEFT  at 512,384  OFFSET 0.13 deg.  4.0,0.7 pix."
    >>> parse_drift_correction_message(msg)
    {'check_x': 512, 'check_y': 384, 'offset_deg': 0.13, ...}
    """
    if pd.isna(drift_msg) or not isinstance(drift_msg, str):
        return None

    try:
        # Pattern: at X,Y  OFFSET deg deg.  x_pix,y_pix pix.
        pattern = r'at\s+(\d+),(\d+)\s+OFFSET\s+([\d.]+)\s+deg\.\s+([-\d.]+),([-\d.]+)\s+pix\.'
        match = re.search(pattern, drift_msg)

        if match:
            check_x = int(match.group(1))
            check_y = int(match.group(2))
            offset_deg = float(match.group(3))
            offset_x_pix = float(match.group(4))
            offset_y_pix = float(match.group(5))

            # Calculate Euclidean distance in pixels
            offset_total_pix = np.sqrt(offset_x_pix**2 + offset_y_pix**2)

            return {
                'check_x': check_x,
                'check_y': check_y,
                'offset_deg': offset_deg,
                'offset_x_pix': offset_x_pix,
                'offset_y_pix': offset_y_pix,
                'offset_total_deg': offset_deg,  # For consistency
                'offset_total_pix': offset_total_pix,
                'drift_status': 'success'  # Assume success if message exists
            }
        else:
            logger.warning(f"Could not parse drift correction message: {drift_msg[:100]}")
            return None

    except Exception as e:
        logger.error(f"Error parsing drift correction message: {e}")
        return None


def parse_validation_message(val_msg: Any) -> Optional[Dict]:
    """
    Parse VALIDATION message from !CAL column to extract calibration metrics.

    The !CAL column contains nested lists with validation summary information
    including quality labels and error statistics.

    Parameters
    ----------
    val_msg : Any
        !CAL message data (may be string representation of nested lists)

    Returns
    -------
    dict or None
        Dictionary with parsed calibration metrics:
        - quality : str - Quality label (GOOD/FAIR/POOR)
        - avg_error_deg : float - Average validation error in degrees
        - max_error_deg : float - Maximum validation error in degrees
        - validation_type : str - Type of validation (e.g., HV9)
        - eye : str - Eye tracked (LEFT/RIGHT/BOTH)
        - offset_deg : float - Offset in degrees (if available)
        - offset_x_pix : float - X offset in pixels (if available)
        - offset_y_pix : float - Y offset in pixels (if available)
        Returns None if parsing fails

    Examples
    --------
    >>> # Example with nested list structure
    >>> msg = "[['VALIDATION', 'HV9', 'L', 'LEFT', '', 'GOOD', 'ERROR', '0.35', 'avg.', '0.52', 'max']]"
    >>> parse_validation_message(msg)
    {'quality': 'GOOD', 'avg_error_deg': 0.35, 'max_error_deg': 0.52, ...}
    """
    if pd.isna(val_msg):
        return None

    try:
        # Parse string to nested lists if needed
        if isinstance(val_msg, str):
            val_data = ast.literal_eval(val_msg)
        else:
            val_data = val_msg

        # Flatten nested structure
        tokens = flatten_nested_lists(val_data)

        # Convert to strings and filter empty
        tokens = [str(t).strip() for t in tokens if str(t).strip()]

        # Find VALIDATION keyword
        if 'VALIDATION' not in tokens:
            return None

        idx_validation = tokens.index('VALIDATION')

        # Extract validation type (e.g., HV9)
        validation_type = tokens[idx_validation + 1] if len(tokens) > idx_validation + 1 else 'UNKNOWN'

        # Extract eye
        eye = 'UNKNOWN'
        if 'LEFT' in tokens:
            eye = 'LEFT'
        elif 'RIGHT' in tokens:
            eye = 'RIGHT'

        # Find quality label (GOOD/FAIR/POOR) - usually before ERROR keyword
        quality = 'UNKNOWN'
        if 'ERROR' in tokens:
            idx_error = tokens.index('ERROR')
            # Quality is typically a few positions before ERROR
            for i in range(max(0, idx_error - 5), idx_error):
                if tokens[i] in ['GOOD', 'FAIR', 'POOR']:
                    quality = tokens[i]
                    break

        # Extract average error (after 'ERROR')
        avg_error = None
        max_error = None
        if 'ERROR' in tokens:
            idx_error = tokens.index('ERROR')
            # Average error is typically right after ERROR
            if len(tokens) > idx_error + 1:
                try:
                    avg_error = float(tokens[idx_error + 1])
                except (ValueError, IndexError):
                    pass

            # Max error is typically after 'avg.' marker
            if 'avg.' in tokens:
                idx_avg = tokens.index('avg.', idx_error)
                if len(tokens) > idx_avg + 1:
                    try:
                        max_error = float(tokens[idx_avg + 1])
                    except (ValueError, IndexError):
                        pass
            elif 'max' in tokens:
                idx_max = tokens.index('max', idx_error)
                if idx_max > 0:
                    try:
                        max_error = float(tokens[idx_max - 1])
                    except (ValueError, IndexError):
                        pass

        # Extract offset information if present
        offset_deg = None
        offset_x_pix = None
        offset_y_pix = None

        if 'OFFSET' in tokens:
            idx_offset = tokens.index('OFFSET')
            if len(tokens) > idx_offset + 1:
                try:
                    offset_deg = float(tokens[idx_offset + 1])
                except (ValueError, IndexError):
                    pass

            # Look for pixel offsets after 'pix.' marker
            if 'pix.' in tokens:
                idx_pix = tokens.index('pix.', idx_offset)
                if idx_pix > 0:
                    # Pixel offsets typically in format "x,y" before 'pix.'
                    pix_str = tokens[idx_pix - 1]
                    if ',' in pix_str:
                        try:
                            parts = pix_str.split(',')
                            offset_x_pix = float(parts[0])
                            offset_y_pix = float(parts[1])
                        except (ValueError, IndexError):
                            pass

        result = {
            'quality': quality,
            'avg_error_deg': avg_error,
            'max_error_deg': max_error,
            'validation_type': validation_type,
            'eye': eye
        }

        # Add optional fields if present
        if offset_deg is not None:
            result['offset_deg'] = offset_deg
        if offset_x_pix is not None:
            result['offset_x_pix'] = offset_x_pix
        if offset_y_pix is not None:
            result['offset_y_pix'] = offset_y_pix

        return result

    except Exception as e:
        logger.error(f"Error parsing validation message: {e}")
        return None


def parse_validate_points(validate_msg: Any) -> Optional[Dict[int, float]]:
    """
    Parse VALIDATE message to extract per-point validation errors.

    The VALIDATE column contains nested lists with individual point measurements
    from the calibration validation.

    Parameters
    ----------
    validate_msg : Any
        VALIDATE message data (may be string representation of nested lists)

    Returns
    -------
    dict or None
        Dictionary mapping point number to error in degrees:
        {0: 0.24, 1: 0.35, 2: 0.28, ...}
        Returns None if parsing fails

    Examples
    --------
    >>> msg = "[['L', 'POINT', '0', '', 'LEFT', '', 'at', '512,384', '', 'OFFSET', '0.24', 'deg.']]"
    >>> parse_validate_points(msg)
    {0: 0.24}
    """
    if pd.isna(validate_msg):
        return None

    try:
        # Parse string to nested lists if needed
        if isinstance(validate_msg, str):
            validate_data = ast.literal_eval(validate_msg)
        else:
            validate_data = validate_msg

        # Flatten nested structure
        tokens = flatten_nested_lists(validate_data)

        # Convert to strings
        tokens = [str(t).strip() for t in tokens if str(t).strip()]

        # Extract point errors
        point_errors = {}
        i = 0
        while i < len(tokens):
            if tokens[i] == 'POINT' and i + 1 < len(tokens):
                try:
                    point_num = int(tokens[i + 1])

                    # Find OFFSET after this point
                    offset_idx = None
                    for j in range(i + 2, min(i + 20, len(tokens))):
                        if tokens[j] == 'OFFSET':
                            offset_idx = j
                            break

                    if offset_idx and offset_idx + 1 < len(tokens):
                        error_deg = float(tokens[offset_idx + 1])
                        point_errors[point_num] = error_deg
                        i = offset_idx + 5
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1

        return point_errors if point_errors else None

    except Exception as e:
        logger.error(f"Error parsing validate points: {e}")
        return None


def extract_validation_events(messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all validation events from messages dataframe.

    Filters rows with non-null !CAL column and parses each validation
    summary to create a dataframe of calibration events.

    Parameters
    ----------
    messages_df : pd.DataFrame
        Messages dataframe with !CAL and !CAL_time columns

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per calibration event containing:
        - timestamp : float
        - event_type : str ('calibration')
        - quality : str
        - avg_error_deg : float
        - max_error_deg : float
        - validation_type : str
        - eye : str
        - Additional columns for point-by-point errors if available
    """
    # Filter to rows with calibration data
    cal_rows = messages_df[messages_df['!CAL'].notna()].copy()

    if len(cal_rows) == 0:
        logger.warning("No calibration events found in messages")
        return pd.DataFrame()

    logger.info(f"Found {len(cal_rows)} calibration events")

    # Parse each calibration message
    parsed_events = []
    for idx, row in cal_rows.iterrows():
        # Parse validation summary from !CAL
        val_dict = parse_validation_message(row['!CAL'])

        if val_dict is None:
            logger.warning(f"Could not parse calibration at index {idx}")
            continue

        # Parse per-point errors from VALIDATE if available
        point_errors = None
        if 'VALIDATE' in row and pd.notna(row['VALIDATE']):
            point_errors = parse_validate_points(row['VALIDATE'])

        # Build event dictionary
        event = {
            'timestamp': row.get('!CAL_time', np.nan),
            'event_type': 'calibration',
            **val_dict
        }

        # Add per-point errors if available
        if point_errors:
            for point_num, error in point_errors.items():
                event[f'point_{point_num}_error'] = error
            event['n_points'] = len(point_errors)

        parsed_events.append(event)

    # Convert to dataframe
    events_df = pd.DataFrame(parsed_events)

    logger.info(f"Successfully parsed {len(events_df)} calibration events")

    return events_df


def extract_drift_events(messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all drift correction events from messages dataframe.

    Filters rows with non-null DRIFTCORRECT column and parses each
    drift correction message.

    Parameters
    ----------
    messages_df : pd.DataFrame
        Messages dataframe with DRIFTCORRECT and DRIFTCORRECT_time columns

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per drift correction event containing:
        - timestamp : float
        - event_type : str ('drift_correction')
        - check_x : float
        - check_y : float
        - offset_deg : float
        - offset_x_pix : float
        - offset_y_pix : float
        - offset_total_deg : float
        - offset_total_pix : float
        - drift_status : str
    """
    # Filter to rows with drift correction data
    drift_rows = messages_df[messages_df['DRIFTCORRECT'].notna()].copy()

    if len(drift_rows) == 0:
        logger.warning("No drift correction events found in messages")
        return pd.DataFrame()

    logger.info(f"Found {len(drift_rows)} drift correction events")

    # Parse each drift correction message
    parsed_events = []
    for idx, row in drift_rows.iterrows():
        drift_dict = parse_drift_correction_message(row['DRIFTCORRECT'])

        if drift_dict is None:
            logger.warning(f"Could not parse drift correction at index {idx}")
            continue

        # Build event dictionary
        event = {
            'timestamp': row.get('DRIFTCORRECT_time', np.nan),
            'event_type': 'drift_correction',
            **drift_dict
        }

        parsed_events.append(event)

    # Convert to dataframe
    events_df = pd.DataFrame(parsed_events)

    logger.info(f"Successfully parsed {len(events_df)} drift correction events")

    return events_df
