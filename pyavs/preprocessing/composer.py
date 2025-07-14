"""
AVS Composer for MEG-ET data fusion in pyAVS package.

This script is used to load the MEG and ET data. It fuses them in the sense that it can
be used to generate ET event based MEG epochs and their related metadata.

Adapted from AVS-machine-room/avs_machine_room/prepro/meg/avs_composer.py
Author(s): P. Sulewski (psulewski@uos.de)
"""

import os
import datetime
from joblib import Parallel, delayed
from typing import List, Dict, Tuple, Optional, Union, Any

import mne
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..dataloader.meg import load_meg_raw, load_meg_preprocessed, load_and_preprocess_meg_run
from ..dataloader.eye import load_and_enrich_eye_events, add_fixation_sequence_position, add_cross_event_information
from ..utils.config import get_data_path
from ..utils.paths import get_subject_session_id, get_max_blocks
from ..utils.validation import validate_subject_id, validate_session
from .trigger_tools import get_meg_trigger_dict, repair_meg_trigger_events, add_fix_event_trigger, get_avs_blocks
from .meg import preprocess_meg_block


class AVSComposer:
    """
    AVS Composer for MEG-ET data alignment and fusion using trigger-based synchronization.
    
    This class implements the complete pipeline for loading, aligning, and fusing MEG and 
    eye-tracking data using the AVS composer approach with scene onset triggers.
    """
    
    def __init__(
        self,
        subject: int,
        session_num: int,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        et_dir: Optional[str] = None,
        diagnostics: Optional[dict] = None,
        preprocessed: bool = True,
        recompute_prepro: bool = False,
        max_block: Optional[int] = None,
        min_block: int = 1,
        stim_channel: str = 'STI101',
        server: str = "uos",
        verbose: bool = True,
        write_output: bool = True,
        interpolate_bad_channels: bool = True,
        n_jobs: int = 1,
        random_state: int = 42
    ):
        """
        Initialize the AVSComposer object.

        Parameters
        ----------
        subject : int
            The subject identifier.
        session_num : int
            The session number.
        data_dir : str, optional
            The directory where MEG and ET data can be found. If None, uses configured data path.
        output_dir : str, optional
            The directory where the output will be saved. If None, uses data_dir.
        et_dir : str, optional
            The directory where the eyetracking data is stored. If None, uses data_dir.
        diagnostics : dict, optional
            A dictionary containing diagnostic information.
        preprocessed : bool, optional
            Whether to run the diagnostics on preprocessed data. Defaults to True.
        recompute_prepro : bool, optional
            Whether to recompute the preprocessed data even if it is available. Defaults to False.
        max_block : int, optional
            The maximum block number. Defaults to None.
        min_block : int, optional
            The minimum block number. Defaults to 1.
        stim_channel : str, optional
            The channel that contains the trigger events. Defaults to 'STI101'.
        server : str, optional
            The server name. Defaults to "uos".
        verbose : bool, optional
            Whether to print additional information. Defaults to True.
        write_output : bool, optional
            Whether to write output to file. Defaults to True.
        interpolate_bad_channels : bool, optional
            Whether to interpolate bad channels. Defaults to True.
        n_jobs : int, optional
            The number of parallel jobs to run. Defaults to 1.
        random_state : int, optional
            The random state. Defaults to 42.
        """
        
        self.subject = validate_subject_id(subject)
        self.session_num = validate_session(session_num)
        self.session = chr(ord('a') + session_num - 1)  # Convert to session letter (1->a, 2->b, etc.)
        
        # Set up data directories
        if data_dir is None:
            data_dir = get_data_path()
            if data_dir is None:
                raise ValueError("No data path configured. Use set_data_path() or provide data_dir parameter")
        
        self.data_dir = data_dir
        self.server = server
        self.et_dir = et_dir if et_dir is not None else data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        
        # Set up block parameters
        self.max_block = max_block
        self.min_block = min_block
        self.blocks_this_session = get_avs_blocks(self.session_num, self.min_block, self.max_block)
        print('Selected blocks this session: ', self.blocks_this_session)
        
        # Set up other parameters
        self.stim_channel = stim_channel
        self.verbose = verbose
        self.sub_sess_id = 'as' + str(self.subject).zfill(2) + self.session
        self.session_dir = os.path.join(self.data_dir, 'rawdir', self.sub_sess_id)
        self.prepro_dir = os.path.join(self.data_dir, 'rawdir', self.sub_sess_id, 'prepro')
        self.subject_dir = os.path.join(self.data_dir, 'rawdir', 'as' + str(self.subject).zfill(2))
        self.write_output = write_output
        self.preprocessed = preprocessed
        self.recompute_prepro = recompute_prepro
        self.interpolate_bad_channels = interpolate_bad_channels
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialize data storage
        self.raws_dict = {}
        self.raws_dict_empty_room = {}
        self.empty_room_recording_names = ['d', 'b']
        self.empty_room_available = False
        
        # Initialize processing state
        self.meg_trigger_events = None
        self.et_events = None
        self.explog = None
        self.raws_concatenated = None
        self.raws_annotated = None
        self.et_epochs = None
        self.et_event_types = None
    
    def read_meg_block(
        self,
        block: Union[int, str],
        compute_missing_prepro: bool = True,
        preload: bool = True,
    ) -> Tuple[Union[int, str], Optional[mne.io.Raw]]:
        """
        Reads the raw data of a given block.

        Parameters
        ----------
        block : int or str
            The block number or empty room recording name.
        compute_missing_prepro : bool, optional
            Flag indicating whether to compute missing preprocessed data. Defaults to True.
        preload : bool, optional
            Flag indicating whether to preload the data. Defaults to True.

        Returns
        -------
        tuple
            A tuple containing the block number and the raw data.
        """
        
        if self.verbose:
            print('Loading data for subject', self.subject, 'session', self.session)
        
        if self.preprocessed:
            fif_suffix = "_raw-sss.fif"
            dir_stem = self.prepro_dir
        else:
            fif_suffix = ".fif"
            dir_stem = self.session_dir
        
        empty_room_recording = False

        # Check if block is from empty room recording
        if block in self.empty_room_recording_names:  # d: danach # b: vorher
            raw_fname = os.path.join(self.prepro_dir, self.sub_sess_id + block + fif_suffix)
            # Check if the file exists
            if not os.path.isfile(raw_fname) and not compute_missing_prepro:
                print('Warning: Empty room recording file not found.', raw_fname)
                return block, None
            empty_room_recording = True
        else:
            raw_fname = os.path.join(dir_stem, self.sub_sess_id + str(block).zfill(2) + fif_suffix)
            print("checking for preprocessed data in: ", raw_fname)

        if os.path.exists(raw_fname) and not self.recompute_prepro:
            print("Found preprocessed data in: ", raw_fname)
            try:
                raw = mne.io.read_raw_fif(
                    raw_fname,
                    preload=preload,
                    verbose=self.verbose
                )
            except Exception as e:
                print(f"Error loading file {raw_fname}: {e}")
                return block, None
        else:
            if self.preprocessed:
                print('No preprocessed raw data found for block', block, raw_fname)
                if compute_missing_prepro or self.recompute_prepro:
                    print('Computing preprocessed data for block', block)
                    if empty_room_recording:
                        raw_fname = os.path.join(self.session_dir, self.sub_sess_id + block + ".fif")
                        # Check whether the file exists
                        if not os.path.isfile(raw_fname):
                            print('Warning: Empty room recording file not found', raw_fname)
                            return block, None
                    else:
                        raw_fname = os.path.join(self.session_dir, self.sub_sess_id + str(block).zfill(2) + ".fif")

                    try:
                        raw_for_recompute = mne.io.read_raw_fif(
                            raw_fname,
                            preload=preload,
                            verbose=self.verbose
                        )
                    except Exception as e:
                        print(f"Error loading raw file for recompute {raw_fname}: {e}")
                        return block, None
                    
                    if empty_room_recording:
                        # We need to prepare the empty room recording for the maxwell filter
                        # For that we need an additional raw file that is not an empty room recording as a reference
                        # We use the first block of the session
                        raw_reference_fname = os.path.join(self.prepro_dir, self.sub_sess_id + str(1).zfill(2) + "_raw-sss.fif")
                        if not os.path.isfile(raw_reference_fname):
                            print('Warning: Preprocessed reference raw file not found. Trying without prepro.', raw_reference_fname)
                            raw_reference_fname = os.path.join(self.session_dir, self.sub_sess_id + str(1).zfill(2) + ".fif")
                            if not os.path.isfile(raw_reference_fname):
                                print('Warning: Reference raw file not found.', raw_reference_fname)
                                return block, None
                        
                        try:
                            raw_reference = mne.io.read_raw_fif(
                                raw_reference_fname,
                                preload=preload,
                                verbose=self.verbose
                            )
                            raw_for_recompute = mne.preprocessing.maxwell_filter_prepare_emptyroom(
                                raw_er=raw_for_recompute,
                                raw=raw_reference,
                                bads='from_raw',
                                annotations='from_raw',
                                meas_date='keep',
                                emit_warning=False,
                                verbose=None
                            )
                        except Exception as e:
                            print(f"Error preparing empty room recording: {e}")
                            return block, None

                    # Apply preprocessing using pyAVS preprocessing function
                    try:
                        raw = preprocess_meg_block(
                            raw_for_recompute,
                            self.subject,
                            self.session_num,
                            block,
                            save_preprocessed=True,
                            data_path=self.data_dir
                        )
                    except Exception as e:
                        print(f"Error preprocessing block {block}: {e}")
                        return block, None
                else:
                    print('Warning: No raw data found for block', block, 'and compute_missing_prepro is set to False')
                    return block, None
            else:
                try:
                    raw = mne.io.read_raw_fif(
                        raw_fname,
                        preload=preload,
                        verbose=self.verbose
                    )
                except Exception as e:
                    print(f"Error loading raw file {raw_fname}: {e}")
                    return block, None

        return block, raw

    def read_meg_sessions(
        self,
        preload: bool = True,
        compute_missing_prepro: bool = False
    ) -> dict:
        """
        Reads the MEG data of a given subject and session per block.
        
        Parameters
        ----------
        preload : bool, optional
            Whether to preload the data. Defaults to True.
        compute_missing_prepro : bool, optional
            Whether to compute missing preprocessing steps. Defaults to False.
        
        Returns
        -------
        dict
            A dictionary containing the raw MEG data for each block.
        """
        
        self.raws_dict = {}
        blocks = list(range(self.min_block, self.max_block + 1))
        # Append the empty room recording blocks
        blocks = blocks + self.empty_room_recording_names
        
        if self.n_jobs == 1:
            for block in blocks:
                # Read the raw data for this block
                block_id, raw = self.read_meg_block(block, compute_missing_prepro=compute_missing_prepro, preload=preload)
                if raw is not None:
                    # TODO: Add bad channel handling from logbook if needed
                    self.raws_dict[block_id] = raw
        else:
            # Use joblib to parallelize the reading of the data
            loading_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.read_meg_block)(block, compute_missing_prepro=compute_missing_prepro, preload=preload) 
                for block in blocks
            )
            for block_id, raw in loading_results:
                if raw is not None:
                    # TODO: Add bad channel handling from logbook if needed
                    self.raws_dict[block_id] = raw
        
        if self.verbose:
            print("Found raw data for blocks", list(self.raws_dict.keys()))
        
        return self.raws_dict

    def load_meg_data(
        self,
        compute_missing_prepro: bool = True,
        preprocessed: bool = True
    ):
        """
        Loads the preprocessed data of a given subject and session per block.
        If the preprocessed data is not available, it will be computed.
        
        Parameters
        ----------
        compute_missing_prepro : bool, optional
            Flag indicating whether to compute missing preprocessed data. Defaults to True.
        preprocessed : bool, optional
            Flag indicating whether to load preprocessed data. Defaults to True.
        """

        print('Loading data for subject', self.subject, 'session', self.session)
        
        if self.max_block is None:
            if self.session == 'a':
                self.max_block = 10
            else:
                self.max_block = 14

        if self.min_block is None:
            self.min_block = 1

        if self.verbose:
            print('Loading data for blocks', self.min_block, 'to', self.max_block)

        raws_dict = self.read_meg_sessions(
            compute_missing_prepro=compute_missing_prepro,
            preload=True
        )
        
        # Check if empty room recordings are available
        self.empty_room_available = False
        for block in self.empty_room_recording_names:
            if block in raws_dict.keys():
                if raws_dict[block] is not None:
                    self.empty_room_available = True
                    self.raws_dict_empty_room[block] = raws_dict[block].copy()
                # Remove the empty room recording from the raws_dict
                del raws_dict[block]

        if self.verbose:
            print('Empty room recordings available: ', self.empty_room_available)
        
        self.raws_dict = raws_dict

    def filter_meg_data(
        self,
        l_freq: float,
        h_freq: float,
        picks=None,
        causal: bool = True,
        concatenated: Optional[bool] = False
    ):
        """
        Applies lowpass and/or highpass filters to the MEG data.

        Parameters
        ----------
        l_freq : float
            The lower frequency cutoff for the filter.
        h_freq : float
            The higher frequency cutoff for the filter.
        picks : list or None, optional
            The indices of the channels to filter. If None, all channels are filtered.
        causal : bool, optional
            Whether to use a causal filter (True) or a non-causal filter (False). Default is True.
        concatenated : bool or None, optional
            Whether to use the concatenated data for filtering. If None, it uses the concatenated data if available, otherwise it uses the data per block.
        """
        
        if self.verbose:
            print('Filtering data for subject', self.subject, 'session', self.session)
        
        # Check if raws have already been concatenated
        if concatenated is None:
            concatenated = hasattr(self, 'raws_concatenated')
        
        phase = 'minimum' if causal else 'zero'
        
        if not concatenated:
            # Filter the data per block
            print("Filtering data per block")
            for block in self.raws_dict.keys():
                self.raws_dict[block].filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    picks=picks,
                    phase=phase,
                    fir_design='firwin',
                    verbose=self.verbose,
                    n_jobs=self.n_jobs
                )
                # We add an attribute that tells us that the data has been filtered
                self.raws_dict[block].filtered = True
        else:
            print("Filtering concatenated data")
            # Filter the concatenated data
            self.raws_concatenated.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                phase=phase,
                fir_design='firwin',
                verbose=self.verbose,
                n_jobs=self.n_jobs
            )
            # We add an attribute that tells us that the data has been filtered
            self.raws_concatenated.filtered = True

    def resample_meg_data(self, target_sfreq: float):
        """
        Resamples the MEG data to the target sampling frequency.

        Parameters
        ----------
        target_sfreq : float
            The target sampling frequency in Hz.
        """
        
        # Check if current sampling frequency is equal to target sampling frequency
        if self.raws_dict[list(self.raws_dict.keys())[0]].info['sfreq'] == target_sfreq:
            print('Data is already sampled at the target sampling frequency')
            return
        
        # Check if we are down sampling or up sampling
        if self.raws_dict[list(self.raws_dict.keys())[0]].info['sfreq'] > target_sfreq:
            print('Downsampling data to', target_sfreq, 'Hz')
        else:
            # This is not recommended
            print('Warning: Upsampling data to', target_sfreq, 'Hz', 'this is not recommended!')

        if self.verbose:
            print('Resampling data for subject', self.subject, 'session', self.session)
        
        # Check if raws have already been concatenated
        concatenated = hasattr(self, 'raws_concatenated')
        
        if not concatenated:
            # Resample the data per block
            for block in self.raws_dict.keys():
                self.raws_dict[block].resample(target_sfreq, n_jobs=self.n_jobs)
                # We add an attribute that tells us that the data has been resampled
                self.raws_dict[block].resampled = True
        else:
            # Resample the concatenated data
            self.raws_concatenated.resample(target_sfreq, n_jobs=self.n_jobs)
            # We add an attribute that tells us that the data has been resampled
            self.raws_concatenated.resampled = True

    def concatenate_raws_per_session(self):
        """
        Concatenates the raws for each block into one raw object per session.
        """
        
        if self.verbose:
            print('Concatenating raws')
        
        raws_list = list(self.raws_dict.values())
        
        # Remove duplicates from the bads list
        for raw in raws_list:
            raw.info['bads'] = list(set(raw.info['bads']))
            
        if self.preprocessed:
            if self.interpolate_bad_channels:
                if self.verbose:
                    print('Interpolating bad channels')
                print("bads: ", [raw.info['bads'] for raw in raws_list])
                raws_list = [raw.interpolate_bads() for raw in raws_list]
            else:
                # We ignore the bad channels labels and hope that the MaxFilter did reasonably well in dealing with them
                # We remove the bad channel info from the raw objects
                for raw in raws_list:
                    raw.info['bads'] = []
        
        self.raws_concatenated = mne.concatenate_raws(raws_list, on_mismatch='warn')
        
        if self.empty_room_available:
            raws_list_empty_room = list(self.raws_dict_empty_room.values())
            if self.preprocessed:
                if self.interpolate_bad_channels:
                    if self.verbose:
                        print('Interpolating bad channels')
                    # Remove duplicates from the bads list
                    for raw in raws_list_empty_room:
                        raw.info['bads'] = list(set(raw.info['bads']))
                    raws_list_empty_room = [raw.interpolate_bads() for raw in raws_list_empty_room]
                else:
                    # We ignore the bad channels labels and hope that the MaxFilter did reasonably well in dealing with them
                    # We remove the bad channel info from the raw objects
                    for raw in raws_list_empty_room:
                        raw.info['bads'] = []
            self.raws_concatenated_empty_room = mne.concatenate_raws(raws_list_empty_room, on_mismatch='warn')

    def find_events_in_raw(self):
        """
        Finds the events in the raw data.
        """
        
        if self.verbose:
            print('Finding events')
        
        self.meg_trigger_events = mne.find_events(
            self.raws_concatenated,
            stim_channel='STI101',
            consecutive=True,
            min_duration=0.008,
            output='onset',
            uint_cast=True
        )

    def get_et_annotations(
        self,
        et_event_types: List[str] = ["fixation", "saccade"],
        recording = "scene",
        exclude_last_fixation: bool = True,
        get_object_labels: bool = False,
        add_cross_event_info: bool = True,
        preprocessed: bool = True,
        save_annotated_raw: bool = False
    ):
        """
        Extracts annotations from the eye tracker data.

        Parameters
        ----------
        et_event_types : list, optional
            List of event types to extract from the eye tracking data. Defaults to ["fixation", "saccade"].
            Valid options: ["fixation", "saccade", "blink"].
        recording : str, optional
            Recording context to filter events by. Defaults to "scene".
            Valid options: ["scene", "caption", "microphone"].
        exclude_last_fixation : bool, optional
            Whether to exclude the last fixation event on each scene. Defaults to True.
        get_object_labels : bool, optional
            Whether to get object labels for the eye tracking events. Defaults to False.
        add_cross_event_info : bool, optional
            Whether to add cross event information to the eye tracking events. Defaults to True.
        preprocessed : bool, optional
            Whether the eye tracking data is preprocessed. Defaults to True.
        save_annotated_raw : bool, optional
            Whether to save the annotated raws. Defaults to False.
        """
        
        # Now we have to read in the eye tracking events
        if self.verbose:
            print('Reading in eye tracking events')

        self.explog, self.et_events = load_and_enrich_eye_events(
            [self.subject],
            [self.session_num],
            data_path=self.et_dir,
            preprocessed=preprocessed,
            fix_multi_saccades=True
        )
        # subselect the recording context 
        if self.verbose:
            print('Subselecting eye tracking events for recording context:', recording)
        assert recording in ["scene", "caption", "microphone"], "Invalid recording context"
        self.et_events = self.et_events[self.et_events["recording"] == recording]
        if self.et_events.empty:
            raise ValueError(f"No eye tracking events found for recording context '{recording}' in subject {self.subject}, session {self.session_num}")
        if add_cross_event_info:
            self.et_events = add_cross_event_information(self.et_events, verbose=self.verbose)

        # Now we will extract the annotations from the eye tracking events
        if self.verbose:
            print('Extracting annotations from eye tracking events')

        self.et_events = add_fixation_sequence_position(self.et_events)
        
        # Now we will remove the last fixation event on each scene
        if exclude_last_fixation:
            if self.verbose:
                print("We removed " + str(len(self.et_events) - len(self.et_events[self.et_events["fix_sequence_from_last"] != 0])) + " fixation events from the eye tracking events, because they were the last fixation event on a scene")
            self.et_events = self.et_events[self.et_events["fix_sequence_from_last"] != 0]

        # Now we add fixation based triggers to the MEG signal
        self.raws_annotated, missing_trials = add_fix_event_trigger(
            self.raws_concatenated,
            blocks=self.blocks_this_session,
            et_events=self.et_events,
            session=self.session_num,
            block_trigger_offset=1000,
            stim_channel='STI101',
            verbose=True,
            event_types=et_event_types,
            recording=recording
        )
        
        # Save the annotated raws
        if save_annotated_raw:
            # Now we will save the annotated raws
            if self.verbose:
                print('Saving annotated raws')
            self.raws_annotated.save(os.path.join(self.output_dir, self.sub_sess_id + '_annotated_raws.fif'), overwrite=True)

        # Print warning that informs about the number of missing trials
        if len(missing_trials) > 0:
            print("Warning: " + str(
                len(missing_trials)) + " trials could not be annotated in the MEG data. They were removed from the eye tracking events. Session: " + str(
                self.session_num))
        
        for missing_trial in missing_trials:
            # Unpack the (block, trial) tuple and remove the respective events from the et_events dataframe
            block, trial = missing_trial
            self.et_events = self.et_events[
                ~((self.et_events["block"] == block) & (self.et_events["trial_per_block"] == trial))]

    def make_et_event_epochs(
        self,
        tmin: float,
        tmax: float, 
        event_types: List[str],
        recording: str = "scene",
        save_epochs: bool = True,
        get_metadata: bool = True,
        get_object_labels: bool = False,
        baseline: Optional[Tuple[float, float]] = None
    ):
        """
        This function will make use of the et event annotations to make event epochs.
        
        Parameters
        ----------
        tmin : float
            The start of the epoch in seconds (around et event onset)
        tmax : float
            The end of the epoch in seconds (around et event onset)
        event_types : list of str
            The event types for which we want to make epochs. E.g. ["fixation", "saccade", "blink"]
            Valid options: ["fixation", "saccade", "blink", "scene"]
        recording : str, optional
            Recording context to filter events by. Defaults to "scene".
            Valid options: ["scene", "caption", "microphone"].
        save_epochs : bool, optional
            Whether to save the epochs to file. Defaults to True.
        get_metadata : bool, optional
            Whether to add metadata to the epochs object. Defaults to True.
        get_object_labels : bool, optional
            Whether to get the object labels for the fixations. Defaults to False.
        baseline : tuple, optional
            The baseline period for the epochs (for AVS currently not recommended). Defaults to None.
        """
    
        # Check whether event_types are valid
        if not set(event_types).issubset(set(["fixation", "saccade", "blink", "scene"])):
            raise ValueError("event_types must be a subset of ['fixation', 'saccade', 'blink', 'scene']")
        else:
            self.et_event_types = event_types

        # Now we will make the event epochs
        if self.verbose:
            print('Making event epochs.')

        # Add the et annotations to the raw data
        # Check if raws_annotated exists:
        if not hasattr(self, "raws_annotated"):
            self.get_et_annotations(et_event_types=event_types, recording=recording, get_object_labels=get_object_labels)

        events_annot = mne.events_from_annotations(
            self.raws_annotated,
            event_id='auto',
            regexp='(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$',
            use_rounding=True,
            chunk_duration=None,
            verbose=self.verbose
        )

        self.et_epochs = {}

        for event_type in event_types:
            # Now we will make the event epochs
            if self.verbose:
                print('Making event epochs for event type:', event_type)
            
            # We will make the epochs
            if event_type == 'fixation':  # An epoch focussing on the fixation period
                event_id = events_annot[1]['fixation']
            elif event_type == 'saccade':  # An epoch focussing on the saccade period
                event_id = events_annot[1]['saccade']
            elif event_type == 'blink':  # An epoch focussing on the blink period
                event_id = events_annot[1]['blink']
            elif event_type == 'scene':  # An epoch focussing on the 4s scene period
                event_id = events_annot[1]['scene']

            events_to_use = events_annot[0][events_annot[0][:, 2] == event_id]
            
            preload = get_metadata  # For metadata the epochs have to be preloaded
            
            if get_metadata:
                metadata = pd.DataFrame(index=np.arange(len(events_to_use)))  # We will prepare an empty dataframe
                # that we will later fill with metadata from the events dataframe
            else:
                metadata = None

            self.et_epochs[event_type] = mne.Epochs(
                self.raws_annotated,
                events_to_use,
                tmin=tmin,
                tmax=tmax,
                preload=preload,
                proj=None,
                event_id={event_type: event_id},
                baseline=baseline,
                metadata=metadata
            )
            
            # Add metadata to the epochs object
            if get_metadata:
                # As metadata we will add all kinds of information about the eye tracking events
                if event_type in ["fixation", "saccade", "blink"]:
                    self.add_et_metadata_to_epochs(metadata_colnames=self.et_events.columns)
                elif event_type == "scene":
                    self.add_scene_metadata_to_epochs(metadata_colnames=self.et_events.columns)

        if self.verbose:
            print("Event epochs created successfully")

    def add_et_metadata_to_epochs(self, metadata_colnames: Optional[List[str]] = None):
        """
        Adds eye tracking metadata to the epochs object.

        Parameters
        ----------
        metadata_colnames : list or None, optional
            List of column names to be added as metadata. If None, all column names available in the events dataframe will be used.
        """
        
        if not hasattr(self, "et_events"):
            raise ValueError("You need to run get_et_annotations first")
        if not hasattr(self, "et_epochs"):
            raise ValueError("You need to run make_et_event_epochs first")

        # Check 1) now we will check whether the events dataframe and the epochs object have the same amount of events
        for et_event_type in self.et_event_types:
            # Preload the epochs object so that we can compute its length
            events_df_for_metadata = self.et_events[self.et_events["type"] == et_event_type]
          
            events_df_for_metadata = events_df_for_metadata[events_df_for_metadata["block"].isin(self.blocks_this_session)]
            
            if metadata_colnames is None:
                # Then we will use all columnnames available.
                metadata_colnames = events_df_for_metadata.columns
            
            if len(events_df_for_metadata) != len(self.et_epochs[et_event_type]):
                print("len events_df_for_metadata: " + str(len(events_df_for_metadata)))
                print("len epochs: " + str(len(self.et_epochs[et_event_type])))
                print(np.unique(events_df_for_metadata.block))
                print(np.unique(events_df_for_metadata.type))
                print(np.unique(events_df_for_metadata.recording))
                # add some diagnostic prints
            
                raise ValueError("The amount of events in the events dataframe and the epochs object are not the same. This should not happen. Please check your data.")
            else:
                # Check 2) now we will check whether the event durations are identical
                durations_from_annot_df = self.et_epochs[et_event_type].annotations.to_data_frame()
                durations_from_annot = durations_from_annot_df.loc[durations_from_annot_df.description == et_event_type, "duration"]
                
                if not np.array_equal(events_df_for_metadata["duration"].values, durations_from_annot.values):
                    print(events_df_for_metadata["duration"].values)
                    print(events_df_for_metadata["duration"])
                    print(self.et_epochs[et_event_type].annotations.duration)
                    raise ValueError("The event durations in the events dataframe and the epochs object are not the same. This should not happen. Please check your data.")
                else:
                    if self.verbose:
                        print("All checks passed. The amount of events and the event durations are identical. We will now add the metadata to the epochs object")
                    
                    # Now we will add the metadata to the epochs object
                    for colname in metadata_colnames:
                        if colname not in self.et_events.columns:
                            print(f"Warning: The column {colname} is not in the events dataframe. Please check your data")
                        else:
                            self.et_epochs[et_event_type].metadata[colname] = events_df_for_metadata[colname].values
            
            print("Metadata added to epochs object")
    
    def add_scene_metadata_to_epochs(self, metadata_colnames: Optional[List[str]] = None):
        """
        This method transforms the et events dataframe into a summary dataframe that has one row per scene.
        It deals with columns like duration, where there are multiple different fixation durations per scene.
        These durations are stored in a list in the respective cell. The method then adds this dataframe to the scene epochs metadata.

        Parameters
        ----------
        metadata_colnames : list, optional
            A list of column names to include in the summary dataframe. If None, all columns will be included.
        """
        
        if not hasattr(self, "et_events"):
            raise ValueError("You need to run get_et_annotations first")
        if not hasattr(self, "et_epochs"):
            raise ValueError("You need to run make_et_event_epochs first")

        # Check get the fixation events
        events_df_for_metadata = self.et_events
        events_df_for_metadata = events_df_for_metadata[events_df_for_metadata["block"].isin(self.blocks_this_session)]
        
        # Now we will transform the et events dataframe into a summary dataframe that has one row per scene
        # We have to keep the order of scenes as stated e.g. by "trial" column
        events_df_for_metadata_grouped = events_df_for_metadata.groupby("trial").agg(lambda x: list(x))
        # Sort by trial
        print(events_df_for_metadata_grouped)
        
        # Now we will add the metadata to the epochs object
        self.et_epochs["scene"].metadata = events_df_for_metadata_grouped
        
        # Add column - time to first event
        self.et_epochs["scene"].metadata["time_to_first_event"] = self.et_epochs["scene"].metadata["time_in_trial"].apply(lambda x: x[0])
        self.et_epochs["scene"].metadata["type_of_first_event"] = self.et_epochs["scene"].metadata["type"].apply(lambda x: x[0])
        
        print("Scene metadata added to epochs object")

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded and processed data.
        
        Returns
        -------
        dict
            Summary information
        """
        summary = {
            'subject': self.subject,
            'session': self.session_num,
            'blocks_loaded': list(self.raws_dict.keys()) if self.raws_dict else [],
            'meg_channels': self.raws_concatenated.info['nchan'] if hasattr(self, 'raws_concatenated') and self.raws_concatenated else 0,
            'meg_samples': len(self.raws_concatenated.times) if hasattr(self, 'raws_concatenated') and self.raws_concatenated else 0,
            'meg_duration': self.raws_concatenated.times[-1] if hasattr(self, 'raws_concatenated') and self.raws_concatenated else 0,
            'eye_events': len(self.et_events) if self.et_events is not None else 0,
            'epochs_created': len(self.et_epochs) if self.et_epochs else 0,
            'annotations': len(self.raws_annotated.annotations) if hasattr(self, 'raws_annotated') and self.raws_annotated else 0,
            'empty_room_available': self.empty_room_available
        }
        
        return summary