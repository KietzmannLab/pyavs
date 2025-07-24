"""
AVS Composer for MEG-ET data fusion in pyAVS package.

This script is used to load the MEG and ET data. It fuses them in the sense that it can
be used to generate ET event based MEG epochs and their related metadata.

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
from ..utils.logging import get_logger
from .trigger_tools import get_meg_trigger_dict, repair_meg_trigger_events, add_fix_event_trigger, get_avs_blocks
from .meg import preprocess_meg_block
from ..io.write import save_annotated_raw


# Initialize logger
logger = get_logger('preprocessing.composer')


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
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        et_path: Optional[str] = None,
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
        apply_ica: bool = False,
        use_precomputed_ica: bool = False,
        ica_solutions_path: Optional[str] = None,
        ica_exclusions_file: Optional[str] = None,
        l_freq: float = 0.2,
        h_freq: float = 100.0,
        resample_freq: float = 500.0,
        causal_filter: bool = False,
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
        data_path : str, optional
            The directory where MEG and ET data can be found. If None, uses configured data path.
        output_path : str, optional
            The directory where the output will be saved. If None, uses data_path.
        et_path : str, optional
            The directory where the eyetracking data is stored. If None, uses data_path.
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
        apply_ica : bool, optional
            Whether to apply ICA for artifact removal during preprocessing. Defaults to False.
        use_precomputed_ica : bool, optional
            Whether to use precomputed ICA solution instead of computing new one. Defaults to False.
        ica_solutions_path : str, optional
            Path to directory containing precomputed ICA solutions. Defaults to None.
        ica_exclusions_file : str, optional
            Path to JSON file containing ICA component exclusions. Defaults to None.
        l_freq : float, optional
            Low-pass frequency in Hz for filtering. Defaults to 0.2.
        h_freq : float, optional
            High-pass frequency in Hz for filtering. Defaults to 100.0.
        resample_freq : float, optional
            Target sampling frequency in Hz for resampling. Defaults to 500.0.
        causal_filter : bool, optional
            Whether to apply causal filtering (preserves temporal order). Defaults to False.
        n_jobs : int, optional
            The number of parallel jobs to run. Defaults to 1.
        random_state : int, optional
            The random state. Defaults to 42.
        """
        
        self.subject = validate_subject_id(subject)
        self.session_num = validate_session(session_num)
        self.session = chr(ord('a') + session_num - 1)  # Convert to session letter (1->a, 2->b, etc.)
        
        # Set up data directories
        if data_path is None:
            data_path = get_data_path()
            if data_path is None:
                raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
        
        self.data_path = data_path
        self.server = server
        self.et_path = et_path if et_path is not None else data_path
        self.output_path = output_path if output_path is not None else data_path
        
        # Set up block parameters
        self.max_block = max_block
        self.min_block = min_block
        self.blocks_this_session = get_avs_blocks(self.session_num, self.min_block, self.max_block)
        logger.info(f'Selected blocks this session: {self.blocks_this_session}')
        
        # Set up other parameters
        self.stim_channel = stim_channel
        self.verbose = verbose
        self.sub_sess_id = 'as' + str(self.subject).zfill(2) + self.session
        self.session_dir = os.path.join(self.data_path, 'rawdir', self.sub_sess_id)
        
        # Use BIDS derivatives directory for preprocessed data  
        self.derivatives_path = os.path.join(self.data_path, 'derivatives', 'pyavs')
        self.prepro_path = os.path.join(self.derivatives_path, f'sub-{self.subject:02d}', f'ses-{self.session_num:02d}', 'meg')
        
        # Ensure derivatives directory exists
        os.makedirs(self.prepro_path, exist_ok=True)
        
        self.subject_dir = os.path.join(self.data_path, 'rawdir', 'as' + str(self.subject).zfill(2))
        self.write_output = write_output
        self.preprocessed = preprocessed
        self.recompute_prepro = recompute_prepro
        self.interpolate_bad_channels = interpolate_bad_channels
        self.apply_ica = apply_ica
        self.use_precomputed_ica = use_precomputed_ica
        self.ica_solutions_path = ica_solutions_path
        self.ica_exclusions_file = ica_exclusions_file
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resample_freq = resample_freq
        self.causal_filter = causal_filter
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
            logger.info(f'Loading data for subject {self.subject}, session {self.session}')
        
        empty_room_recording = False

        # Check if block is from empty room recording  
        if block in self.empty_room_recording_names:  # d: danach # b: vorher
            empty_room_recording = True
            if self.preprocessed:
                # BIDS-compliant filename for empty room in derivatives with task-noise_meg
                meg_filename = f"sub-{self.subject:02d}_ses-{self.session_num:02d}_task-noise_recording-{block}_raw-sss.fif"
                raw_fname = os.path.join(self.prepro_path, meg_filename)
            else:
                raw_fname = os.path.join(self.session_dir, self.sub_sess_id + block + ".fif")
        else:
            if self.preprocessed:
                # BIDS-compliant filename for regular blocks in derivatives
                meg_filename = f"sub-{self.subject:02d}_ses-{self.session_num:02d}_task-avs_run-{block:02d}_raw-sss.fif"
                raw_fname = os.path.join(self.prepro_path, meg_filename)
            else:
                raw_fname = os.path.join(self.session_dir, self.sub_sess_id + str(block).zfill(2) + ".fif")
        
        logger.debug(f"Checking for data in: {raw_fname}")

        if os.path.exists(raw_fname) and not self.recompute_prepro:
            logger.info(f"Found preprocessed data in: {raw_fname}")
            try:
                raw = mne.io.read_raw_fif(
                    raw_fname,
                    preload=preload,
                    verbose=self.verbose
                )
            except Exception as e:
                logger.error(f"Error loading file {raw_fname}: {e}")
                return block, None
        else:
            if self.preprocessed:
                logger.warning(f'No preprocessed raw data found for block {block}: {raw_fname}')
                if compute_missing_prepro or self.recompute_prepro:
                    logger.info(f'Computing preprocessed data for block {block}')
                    
                    # Use the standardized MEG dataloader for preprocessing
                    try:
                        if not empty_room_recording:
                            # For regular blocks, use the standardized preprocessing
                            raw_for_recompute = load_and_preprocess_meg_run(
                                self.subject, 
                                self.session_num, 
                                block,
                                data_path=self.data_path,
                                force_recompute=True,
                                save_preprocessed=True
                            )
                            return block, raw_for_recompute
                        else:
                            # For empty room recordings, load raw data
                            raw_fname = os.path.join(self.session_dir, self.sub_sess_id + block + ".fif")
                            logger.info(f"Loading empty room recording '{block}' from: {raw_fname}")
                            if not os.path.isfile(raw_fname):
                                logger.error(f'Empty room recording file not found: {raw_fname}')
                                return block, None
                            
                            raw_for_recompute = mne.io.read_raw_fif(
                                raw_fname,
                                preload=preload,
                                verbose=self.verbose
                            )
                    except Exception as e:
                        logger.error(f"Error loading/preprocessing data for block {block}: {e}")
                        return block, None
                    
                    if empty_room_recording:
                        # For empty room recordings, apply preprocessing with proper preparation
                        logger.info(f"Processing empty room recording '{block}' for subject {self.subject}, session {self.session_num}")
                        try:
                            from .meg import preprocess_meg_block
                            # Prepare empty room recording properly if needed
                            ref_filename = f"sub-{self.subject:02d}_ses-{self.session_num:02d}_task-avs_run-01_raw-sss.fif"
                            raw_reference_fname = os.path.join(self.prepro_path, ref_filename)
                            logger.debug(f"Looking for reference file: {raw_reference_fname}")
                            
                            if os.path.isfile(raw_reference_fname):
                                logger.info(f"Using reference file for empty room preparation: {ref_filename}")
                                raw_reference = mne.io.read_raw_fif(raw_reference_fname, preload=preload, verbose=self.verbose)
                                from .meg import prepare_empty_room_recording
                                logger.info(f"Preparing empty room recording '{block}' using reference data")
                                raw_for_recompute = prepare_empty_room_recording(
                                    raw_empty_room=raw_for_recompute,
                                    raw_reference=raw_reference,
                                    bads='from_raw',
                                    annotations='from_raw',
                                    meas_date='keep',
                                    verbose=self.verbose
                                )
                            else:
                                logger.warning(f"No reference file found for empty room preparation: {raw_reference_fname}")
                                logger.info(f"Processing empty room recording '{block}' without reference preparation")
                            
                            # Apply preprocessing to empty room data
                            logger.info(f"Applying preprocessing to empty room recording '{block}'")
                            raw = preprocess_meg_block(
                                raw_for_recompute,
                                subject_id=self.subject,
                                session=self.session_num,
                                block=block,
                                l_freq=self.l_freq,
                                h_freq=self.h_freq,
                                resample_freq=self.resample_freq,
                                causal_filter=self.causal_filter,
                                verbose=self.verbose
                            )
                            logger.info(f"Successfully preprocessed empty room recording '{block}'")
                            
                            # Save the preprocessed empty room data with BIDS-compliant task-noise naming
                            meg_filename = f"sub-{self.subject:02d}_ses-{self.session_num:02d}_task-noise_recording-{block}_raw-sss.fif"
                            output_path = os.path.join(self.prepro_path, meg_filename)
                            raw.save(output_path, overwrite=True)
                            logger.info(f"Saved preprocessed empty room recording to: {output_path}")
                        except Exception as e:
                            logger.error(f"Error processing empty room recording '{block}': {str(e)}")
                            logger.debug(f"Empty room processing error details:", exc_info=True)
                            return block, None
                else:
                    logger.warning(f'No raw data found for block {block} and compute_missing_prepro is set to False')
                    return block, None
            else:
                try:
                    raw = mne.io.read_raw_fif(
                        raw_fname,
                        preload=preload,
                        verbose=self.verbose
                    )
                except Exception as e:
                    logger.error(f"Error loading raw file {raw_fname}: {e}")
                    return block, None
    
        logger.info(f'Load raw data for block {block} from {raw_fname}')
        print(raw)
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
            logger.info(f"Found raw data for blocks: {list(self.raws_dict.keys())}")
        
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

        logger.info(f'Loading data for subject {self.subject}, session {self.session}')
        
        if self.max_block is None:
            if self.session == 'a':
                self.max_block = 10
            else:
                self.max_block = 14

        if self.min_block is None:
            self.min_block = 1

        if self.verbose:
            logger.info(f'Loading data for blocks {self.min_block} to {self.max_block}')

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
            logger.info(f'Empty room recordings available: {self.empty_room_available}')
        
        self.raws_dict = raws_dict

    def filter_meg_data(
        self,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        picks=None,
        causal: Optional[bool] = None,
        concatenated: Optional[bool] = False
    ):
        """
        Applies lowpass and/or highpass filters to the MEG data using pyAVS filter_meg function.

        Parameters
        ----------
        l_freq : float, optional
            The lower frequency cutoff for the filter. If None, uses instance variable.
        h_freq : float, optional
            The higher frequency cutoff for the filter. If None, uses instance variable.
        picks : list or None, optional
            The indices of the channels to filter. If None, all channels are filtered.
        causal : bool, optional
            Whether to use a causal filter. If None, uses instance variable.
        concatenated : bool or None, optional
            Whether to use the concatenated data for filtering. If None, it uses the concatenated data if available, otherwise it uses the data per block.
            
        Notes
        -----
        WARNING: If recompute_prepro=True was used during initialization, the data may already be filtered 
        by preprocess_meg_block. This method should only be used for additional filtering or when 
        recompute_prepro=False.
        """
        from .meg import filter_meg
        
        # Use instance variables as defaults
        if l_freq is None:
            l_freq = self.l_freq
        if h_freq is None:
            h_freq = self.h_freq
        if causal is None:
            causal = self.causal_filter
        
        if self.verbose:
            logger.info(f'Filtering data for subject {self.subject}, session {self.session}')
            
        # Check if data is already filtered by examining MNE info
        if concatenated and hasattr(self, 'raws_concatenated'):
            raw_to_check = self.raws_concatenated
        elif self.raws_dict:
            raw_to_check = list(self.raws_dict.values())[0]
        else:
            raw_to_check = None
            
        if raw_to_check and self.verbose:
            # Check for existing filters in MNE info
            if raw_to_check.info.get('lowpass') is not None:
                logger.info(f'   Data already has lowpass filter at {raw_to_check.info["lowpass"]} Hz')
            if raw_to_check.info.get('highpass') is not None:
                logger.info(f'   Data already has highpass filter at {raw_to_check.info["highpass"]} Hz')
            
            # Additional warning if recompute_prepro was used
            if self.recompute_prepro:
                logger.warning('   Data may already be filtered by preprocess_meg_block during initialization')
        
        # Check if raws have already been concatenated
        if concatenated is None:
            concatenated = hasattr(self, 'raws_concatenated')
        
        if not concatenated:
            # Filter the data per block using meg.py function
            logger.info("Filtering data per block")
            for block in self.raws_dict.keys():
                self.raws_dict[block] = filter_meg(
                    self.raws_dict[block],
                    l_freq=l_freq,
                    h_freq=h_freq,
                    picks=picks,
                    causal=causal,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                # We add an attribute that tells us that the data has been filtered
                self.raws_dict[block].filtered = True
                print(self.raws_dict[block])
        else:
            logger.info("Filtering concatenated data")
            # Filter the concatenated data using meg.py function
            self.raws_concatenated = filter_meg(
                self.raws_concatenated,
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                causal=causal,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            # We add an attribute that tells us that the data has been filtered
            self.raws_concatenated.filtered = True

    def resample_meg_data(self, target_sfreq: Optional[float] = None):
        """
        Resamples the MEG data to the target sampling frequency using pyAVS resample_meg function.

        Parameters
        ----------
        target_sfreq : float, optional
            The target sampling frequency in Hz. If None, uses instance variable.
            
        Notes
        -----
        WARNING: If recompute_prepro=True was used during initialization, the data may already be resampled 
        by preprocess_meg_block. This method should only be used for additional resampling or when 
        recompute_prepro=False.
        """
        from .meg import resample_meg
        
        # Use instance variable as default
        if target_sfreq is None:
            target_sfreq = self.resample_freq
        
        # Check if current sampling frequency is equal to target sampling frequency
        if self.raws_dict[list(self.raws_dict.keys())[0]].info['sfreq'] == target_sfreq:
            logger.info('Data is already sampled at the target sampling frequency')
            return
        
        # Check if we are down sampling or up sampling
        if self.raws_dict[list(self.raws_dict.keys())[0]].info['sfreq'] > target_sfreq:
            logger.info(f'Downsampling data to {target_sfreq} Hz')
        else:
            # This is not recommended
            logger.warning(f'Upsampling data to {target_sfreq} Hz - this is not recommended!')

        if self.verbose:
            logger.info(f'Resampling data for subject {self.subject}, session {self.session}')
            
            # Check current sampling frequency from MNE info
            current_sfreq = self.raws_dict[list(self.raws_dict.keys())[0]].info['sfreq']
            logger.info(f'   Current sampling frequency: {current_sfreq} Hz')
            logger.info(f'   Target sampling frequency: {target_sfreq} Hz')
            
            # Additional warning if recompute_prepro was used
            if self.recompute_prepro:
                logger.warning('   Data may already be resampled by preprocess_meg_block during initialization')
        
        # Check if raws have already been concatenated
        concatenated = hasattr(self, 'raws_concatenated')
        
        if not concatenated:
            # Resample the data per block using meg.py function
            for block in self.raws_dict.keys():
                self.raws_dict[block] = resample_meg(
                    self.raws_dict[block],
                    sfreq=target_sfreq,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                # We add an attribute that tells us that the data has been resampled
                self.raws_dict[block].resampled = True
        else:
            # Resample the concatenated data using meg.py function
            self.raws_concatenated = resample_meg(
                self.raws_concatenated,
                sfreq=target_sfreq,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            # We add an attribute that tells us that the data has been resampled
            self.raws_concatenated.resampled = True

    def concatenate_raws_per_session(self):
        """
        Concatenates the raws for each block into one raw object per session.
        """
        
        if self.verbose:
            logger.info('Concatenating raws')
        
        raws_list = list(self.raws_dict.values())
        
        # Remove duplicates from the bads list
        for raw in raws_list:
            raw.info['bads'] = list(set(raw.info['bads']))
            
        if self.preprocessed:
            if self.interpolate_bad_channels:
                from .meg import interpolate_bad_channels
                if self.verbose:
                    logger.info('Interpolating bad channels')
                logger.debug(f"bads: {[raw.info['bads'] for raw in raws_list]}")
                raws_list = [interpolate_bad_channels(raw, verbose=self.verbose) for raw in raws_list]
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
                    from .meg import interpolate_bad_channels
                    if self.verbose:
                        logger.info('Interpolating bad channels in empty room data')
                    # Remove duplicates from the bads list
                    for raw in raws_list_empty_room:
                        raw.info['bads'] = list(set(raw.info['bads']))
                    raws_list_empty_room = [interpolate_bad_channels(raw, verbose=self.verbose) for raw in raws_list_empty_room]
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
            logger.info('Finding events')
        
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
        event_type: str = "fixation",
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
        et_event_type : str, optional
            Event type to extract from the eye tracking data. Defaults to "fixation".
            Valid options: "fixation", "saccade", "blink".
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
            logger.info('Reading in eye tracking events')

        self.explog, self.et_events = load_and_enrich_eye_events(
            [self.subject],
            [self.session_num],
            data_path=self.et_path,
            preprocessed=preprocessed,
            fix_multi_saccades=True
        )
        # subselect the recording context 
        if self.verbose:
            logger.info(f'Subselecting eye tracking events for recording context: {recording}')
        assert recording in ["scene", "caption", "microphone"], "Invalid recording context"
        self.et_events = self.et_events[self.et_events["recording"] == recording]
        if self.et_events.empty:
            raise ValueError(f"No eye tracking events found for recording context '{recording}' in subject {self.subject}, session {self.session_num}")
        if add_cross_event_info:
            self.et_events = add_cross_event_information(self.et_events, verbose=self.verbose)

        # Now we will extract the annotations from the eye tracking events
        if self.verbose:
            logger.info('Extracting annotations from eye tracking events')

        self.et_events = add_fixation_sequence_position(self.et_events)
        
        # Now we will remove the last fixation event on each scene
        if exclude_last_fixation:
            if self.verbose:
                removed_count = len(self.et_events) - len(self.et_events[self.et_events["fix_sequence_from_last"] != 0])
                logger.info(f"Removed {removed_count} fixation events because they were the last fixation event on a scene")
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
            event_types=[et_event_type],
            recording=recording
        )
        
        # Save the annotated raws
        if save_annotated_raw:
            # Now we will save the annotated raws to the derivatives/annotated directory
            if self.verbose:
                logger.info('Saving annotated raws')
            save_annotated_raw(self.raws_annotated, self.subject, self.session_num, data_path=self.data_path)

        # Print warning that informs about the number of missing trials
        if len(missing_trials) > 0:
            logger.warning(f"{len(missing_trials)} trials could not be annotated in the MEG data. They were removed from the eye tracking events. Session: {self.session_num}")
        
        for missing_trial in missing_trials:
            # Unpack the (block, trial) tuple and remove the respective events from the et_events dataframe
            block, trial = missing_trial
            self.et_events = self.et_events[
                ~((self.et_events["block"] == block) & (self.et_events["trial_per_block"] == trial))]

    def make_et_event_epochs(
        self,
        tmin: float,
        tmax: float, 
        event_type: str,
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
        event_type : str
            The event type for which we want to make epochs. E.g. "fixation", "saccade", "blink"
            Valid options: "fixation", "saccade", "blink", "scene"
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
    
        # Check whether event_type is valid
        if event_type not in ["fixation", "saccade", "blink", "scene"]:
            raise ValueError("event_type must be one of ['fixation', 'saccade', 'blink', 'scene']")
        else:
            self.et_event_type = event_type

        # Now we will make the event epochs
        if self.verbose:
            logger.info('Making event epochs.')

        # Add the et annotations to the raw data
        # Check if raws_annotated exists:
        if not hasattr(self, "raws_annotated"):
            self.get_et_annotations(et_event_type=event_type, recording=recording, get_object_labels=get_object_labels)

        events_annot = mne.events_from_annotations(
            self.raws_annotated,
            event_id='auto',
            regexp='(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$',
            use_rounding=True,
            chunk_duration=None,
            verbose=self.verbose
        )

        # Now we will make the event epochs
        if self.verbose:
            logger.info(f'Making event epochs for event type: {event_type}')
        
        # We will make the epochs
        available_events = events_annot[1]
        
        if event_type not in available_events:
            available_event_types = list(available_events.keys())
            raise ValueError(f"Event type '{event_type}' not found in annotations. "
                           f"Available event types: {available_event_types}")
        
        event_id = available_events[event_type]

        events_to_use = events_annot[0][events_annot[0][:, 2] == event_id]
        
        preload = get_metadata  # For metadata the epochs have to be preloaded
        
        if get_metadata:
            metadata = pd.DataFrame(index=np.arange(len(events_to_use)))  # We will prepare an empty dataframe
            # that we will later fill with metadata from the events dataframe
        else:
            metadata = None

        self.et_epochs = mne.Epochs(
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
            logger.info("Event epochs created successfully")

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
        et_event_type = self.et_event_type
        # Preload the epochs object so that we can compute its length
        events_df_for_metadata = self.et_events[self.et_events["type"] == et_event_type]
      
        events_df_for_metadata = events_df_for_metadata[events_df_for_metadata["block"].isin(self.blocks_this_session)]
        
        if metadata_colnames is None:
            # Then we will use all columnnames available.
            metadata_colnames = events_df_for_metadata.columns
        
        if len(events_df_for_metadata) != len(self.et_epochs):
            logger.error(f"len events_df_for_metadata: {len(events_df_for_metadata)}")
            logger.error(f"len epochs: {len(self.et_epochs)}")
            logger.debug(f"Unique blocks: {np.unique(events_df_for_metadata.block)}")
            logger.debug(f"Unique types: {np.unique(events_df_for_metadata.type)}")
            logger.debug(f"Unique recordings: {np.unique(events_df_for_metadata.recording)}")
            # add some diagnostic prints
        
            raise ValueError("The amount of events in the events dataframe and the epochs object are not the same. This should not happen. Please check your data.")
        else:
            # Check 2) now we will check whether the event durations are identical
            durations_from_annot_df = self.et_epochs.annotations.to_data_frame()
            durations_from_annot = durations_from_annot_df.loc[durations_from_annot_df.description == et_event_type, "duration"]
            
            if not np.array_equal(events_df_for_metadata["duration"].values, durations_from_annot.values):
                logger.debug(f"Events dataframe duration values: {events_df_for_metadata['duration'].values}")
                logger.debug(f"Events dataframe duration column: {events_df_for_metadata['duration']}")
                logger.debug(f"Epochs annotations duration: {self.et_epochs.annotations.duration}")
                raise ValueError("The event durations in the events dataframe and the epochs object are not the same. This should not happen. Please check your data.")
            else:
                if self.verbose:
                    logger.info("All checks passed. The amount of events and the event durations are identical. Adding metadata to the epochs object")
                
                # Now we will add the metadata to the epochs object
                for colname in metadata_colnames:
                    if colname not in self.et_events.columns:
                        logger.warning(f"The column {colname} is not in the events dataframe. Please check your data")
                    else:
                        self.et_epochs.metadata[colname] = events_df_for_metadata[colname].values
        
        logger.info("Metadata added to epochs object")
    
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
        logger.debug(f"Grouped metadata: {events_df_for_metadata_grouped}")
        
        # Now we will add the metadata to the epochs object
        self.et_epochs.metadata = events_df_for_metadata_grouped
        
        # Add column - time to first event
        self.et_epochs.metadata["time_to_first_event"] = self.et_epochs.metadata["time_in_trial"].apply(lambda x: x[0])
        self.et_epochs.metadata["type_of_first_event"] = self.et_epochs.metadata["type"].apply(lambda x: x[0])
        
        logger.info("Scene metadata added to epochs object")

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
            'epochs_created': len(self.et_epochs) if hasattr(self, 'et_epochs') and self.et_epochs else 0,
            'annotations': len(self.raws_annotated.annotations) if hasattr(self, 'raws_annotated') and self.raws_annotated else 0,
            'empty_room_available': self.empty_room_available
        }
        
        return summary

    def apply_ica_to_blocks(self,
                           use_precomputed: Optional[bool] = None,
                           compute_new_ica: bool = False,
                           find_artifacts: bool = True) -> None:
        """
        Apply ICA artifact removal to loaded MEG blocks.
        
        This method applies ICA (either precomputed or newly computed) to remove
        artifacts from the loaded raw MEG data blocks. It operates on unconcatenated
        data for optimal artifact removal.
        
        Parameters
        ----------
        use_precomputed : bool, optional
            Whether to use precomputed ICA solutions. If None, uses instance variable.
        compute_new_ica : bool, optional
            Whether to compute new ICA if precomputed not available (default: False)
        find_artifacts : bool, optional
            Whether to automatically find artifact components when computing new ICA (default: True)
            
        Notes
        -----
        This method modifies self.raws_dict in place with ICA-cleaned data.
        ICA is applied before concatenation for optimal results. The usage of (precomputed) ica implies the interpolation of bad channels.
        """
        from .ica import apply_ica_to_raws
        
        if not self.raws_dict:
            if self.verbose:
                logger.warning("No raw data loaded. Please run load_meg_data() first.")
            return
        
        # Use instance variables as defaults
        if use_precomputed is None:
            use_precomputed = self.use_precomputed_ica
        
        if self.verbose:
            logger.info("Applying ICA artifact removal to MEG blocks...")
            
        if self.verbose:
            logger.info(f"We will have to check whether the bad channels have been interpolated. If not, we will interpolate them now.")
        # Check if bad channels have been interpolated
        if not hasattr(self, 'raws_dict_interpolated') or not self.raws_dict_interpolated:
            if self.verbose:
                logger.info("Interpolating bad channels in raw data before applying ICA")
            from .meg import interpolate_bad_channels
            # Interpolate bad channels for each raw in raws_dict
            self.raws_dict_interpolated = {
                block_id: interpolate_bad_channels(raw, verbose=self.verbose)
                for block_id, raw in self.raws_dict.items()
            }   
        else:
            if self.verbose:
                logger.info("Bad channels have already been interpolated in the raw data")
        # Use the interpolated raws_dict for ICA
        self.raws_dict = self.raws_dict_interpolated
     
        # Apply ICA using the standalone function
        self.raws_dict = apply_ica_to_raws(
            raws_dict=self.raws_dict,
            subject_id=self.subject,
            session=self.session_num,
            use_precomputed=use_precomputed,
            ica_solutions_dir=self.ica_solutions_path,
            ica_exclusions_file=self.ica_exclusions_file,
            compute_new_ica=compute_new_ica,
            find_artifacts=find_artifacts,
            verbose=self.verbose
        )
        
        if self.verbose:
            logger.info("ICA artifact removal completed for all blocks")