"""
Eye tracking utilities for pyAVS package.

This module provides utilities for processing and analyzing eye tracking data,
including functions for matching saccades to fixations and extracting temporal
relationships between eye movement events.
"""

import pandas as pd
import numpy as np
from typing import Literal
from ..utils.logging import get_logger

logger = get_logger('utils.eye_tracking')


def match_saccades_to_fixations(
    saccades_meta_df: pd.DataFrame,
    fixations_meta_df: pd.DataFrame,
    saccade_type: Literal["pre-saccade", "post-saccade"] = "pre-saccade"
) -> pd.DataFrame:
    """
    Match saccades to fixations based on temporal adjacency.

    This function identifies saccade-fixation pairs by analyzing the temporal
    sequence of events within each scene. It matches events that occur
    consecutively with zero time gap between them.

    Parameters
    ----------
    saccades_meta_df : pd.DataFrame
        Metadata for saccades. Must contain columns: 'sceneID', 'type',
        'start_time', 'end_time', 'duration'
    fixations_meta_df : pd.DataFrame
        Metadata for fixations. Must contain columns: 'sceneID', 'type',
        'start_time', 'end_time', 'duration', 'fix_sequence'
    saccade_type : Literal["pre-saccade", "post-saccade"], default="pre-saccade"
        Type of matching to perform:
        - "pre-saccade": Match saccade -> fixation sequences
        - "post-saccade": Match fixation -> saccade sequences

    Returns
    -------
    pd.DataFrame
        Matched saccades with associated fixation information. Includes all
        original saccade columns plus:
        - 'associated_fix_sequence': Sequence number of matched fixation
        - 'associated_fix_start_time': Start time of matched fixation
        - 'associated_fixation_duration': Duration of matched fixation

    Notes
    -----
    Only pairs with exactly 0 time difference between consecutive events are
    included (i.e., saccade.end_time == fixation.start_time for pre-saccade,
    or fixation.end_time == saccade.start_time for post-saccade).

    Examples
    --------
    >>> # Match saccades to subsequent fixations
    >>> matched_df = match_saccades_to_fixations(
    ...     saccades_df, fixations_df, saccade_type="pre-saccade"
    ... )
    >>>
    >>> # Access matched fixation durations
    >>> fixation_durations = matched_df['associated_fixation_duration']
    """
    logger.info(f"Matching saccades to fixations ({saccade_type})...")

    # Combine and sort by time within scenes
    combined_df = pd.concat([fixations_meta_df, saccades_meta_df], axis=0)

    selected_saccades_rows = []
    time_differences = []
    num_events_with_0_time_difference = 0

    unique_sceneIDs = saccades_meta_df['sceneID'].unique()

    for sceneID in unique_sceneIDs:
        scene_group = combined_df[combined_df['sceneID'] == sceneID]
        sorted_group = scene_group.sort_values(by='start_time')

        types = sorted_group['type'].values

        for i in range(len(types) - 1):
            if saccade_type == "pre-saccade":
                # Match: saccade -> fixation
                if types[i] == "saccade" and types[i + 1] == "fixation":
                    saccade_end_time = sorted_group.iloc[i]['end_time']
                    fixation_start_time = sorted_group.iloc[i + 1]['start_time']

                    time_difference = fixation_start_time - saccade_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i].to_dict()
                        saccade_row_data['original_index'] = sorted_group.index[i]
                        saccade_row_data['associated_fix_sequence'] = sorted_group.iloc[i + 1]['fix_sequence']
                        saccade_row_data['associated_fix_start_time'] = sorted_group.iloc[i + 1]['start_time']
                        saccade_row_data['associated_fixation_duration'] = sorted_group.iloc[i + 1]['duration']
                        selected_saccades_rows.append(saccade_row_data)

            elif saccade_type == "post-saccade":
                # Match: fixation -> saccade
                if types[i] == "fixation" and types[i + 1] == "saccade":
                    fixation_end_time = sorted_group.iloc[i]['end_time']
                    saccade_start_time = sorted_group.iloc[i + 1]['start_time']

                    time_difference = saccade_start_time - fixation_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i + 1].to_dict()
                        saccade_row_data['original_index'] = sorted_group.index[i + 1]
                        saccade_row_data['associated_fix_sequence'] = sorted_group.iloc[i]['fix_sequence']
                        saccade_row_data['associated_fix_start_time'] = sorted_group.iloc[i]['start_time']
                        saccade_row_data['associated_fixation_duration'] = sorted_group.iloc[i]['duration']
                        selected_saccades_rows.append(saccade_row_data)

    selected_saccades_df = pd.DataFrame(selected_saccades_rows)
    if len(selected_saccades_df) > 0:
        selected_saccades_df.set_index('original_index', inplace=True)

    logger.info(f"Matched {len(selected_saccades_df)} saccade-fixation pairs")
    logger.info(f"Events with 0 time difference: {num_events_with_0_time_difference}")

    return selected_saccades_df
