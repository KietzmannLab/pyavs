"""
Caption embedding functions for pyAVS.

This module provides functions to encode captions into embeddings using various language models.
Default model is multilingual BERT for cross-language support.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from ..utils.logging import get_logger
import torch
logger = get_logger('captions.embedding')

# Optional dependencies - will be imported when needed
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


def encode_captions(captions: Union[List[str], pd.Series],
                   model_name: str = 'distiluse-base-multilingual-cased',
                   model_type: str = 'sentence-transformers',
                   batch_size: int = 32,
                   max_length: int = 512,
                   device: Optional[str] = None,
                   return_tensors: bool = False) -> np.ndarray:
    """
    Encode captions into embeddings using specified language model.
    
    Parameters
    ----------
    captions : list of str or pd.Series
        Captions to encode
    model_name : str, default 'distiluse-base-multilingual-cased'
        Model name/path. Options:
        - 'distiluse-base-multilingual-cased' (default, fast multilingual)
        - 'sentence-transformers/all-MiniLM-L12-v2' (English)
        - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
        - Any sentence-transformers or HuggingFace model name
    model_type : str, default 'sentence-transformers'
        Type of model loading: 'sentence-transformers' or 'transformers'
    batch_size : int, default 32
        Batch size for encoding
    max_length : int, default 512
        Maximum sequence length
    device : str, optional
        Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None.
    return_tensors : bool, default False
        Return torch tensors instead of numpy arrays
        
    Returns
    -------
    np.ndarray or torch.Tensor
        Embeddings array of shape (n_captions, embedding_dim)
    """
    if isinstance(captions, pd.Series):
        captions = captions.tolist()
    
    # Filter out None/NaN captions
    valid_captions = []
    valid_indices = []
    for i, caption in enumerate(captions):
        if caption is not None and str(caption).strip() and str(caption) != 'nan':
            valid_captions.append(str(caption).strip())
            valid_indices.append(i)
    
    if not valid_captions:
        logger.warning("No valid captions to encode")
        return np.array([])
    
    logger.info(f"Encoding {len(valid_captions)} captions with {model_name}")
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps' 
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    if model_type == 'sentence-transformers':
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        embeddings = _encode_with_sentence_transformers(
            valid_captions, model_name, batch_size, device
        )
    else:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")
        
        embeddings = _encode_with_transformers(
            valid_captions, model_name, batch_size, max_length, device
        )
    
    # Create full embedding array with zeros for invalid captions
    if len(valid_indices) < len(captions):
        embedding_dim = embeddings.shape[1]
        full_embeddings = np.zeros((len(captions), embedding_dim))
        full_embeddings[valid_indices] = embeddings
        embeddings = full_embeddings
    
    if return_tensors and isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    elif not return_tensors and torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def _encode_with_sentence_transformers(captions: List[str], model_name: str, 
                                     batch_size: int, device: str) -> np.ndarray:
    """Encode captions using sentence-transformers."""
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(captions, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def _encode_with_transformers(captions: List[str], model_name: str, 
                            batch_size: int, max_length: int, device: str) -> np.ndarray:
    """Encode captions using transformers library."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_captions,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Get model outputs
            outputs = model(**inputs)
            
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0).numpy()


def encode_caption_dataframe(df: pd.DataFrame,
                           caption_columns: List[str] = ['transcribed_caption'],
                           model_name: str = 'distiluse-base-multilingual-cased',
                           model_type: str = 'sentence-transformers',
                           batch_size: int = 32,
                           max_length: int = 512,
                           device: Optional[str] = None,
                           suffix: str = '_embedding') -> pd.DataFrame:
    """
    Encode caption columns in a DataFrame and add embedding columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing caption columns
    caption_columns : list of str, default ['transcribed_caption']
        Column names containing captions to encode
    model_name : str, default 'bert-base-multilingual-cased'
        Model name for encoding
    model_type : str, default 'transformers'
        Type of model loading
    batch_size : int, default 32
        Batch size for encoding
    max_length : int, default 512
        Maximum sequence length
    device : str, optional
        Device to use
    suffix : str, default '_embedding'
        Suffix to add to embedding column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional embedding columns
    """
    df_copy = df.copy()
    
    for col in caption_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        logger.info(f"Encoding column: {col}")
        
        embeddings = encode_captions(
            captions=df[col],
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            max_length=max_length,
            device=device
        )
        
        # Add embeddings as new column
        embedding_col_name = f"{col}{suffix}"
        df_copy[embedding_col_name] = embeddings.tolist()
        
        logger.info(f"Added embedding column: {embedding_col_name}")
    
    return df_copy


def encode_mscoco_captions(df: pd.DataFrame,
                         mscoco_column: str = 'mscoco_captions',
                         model_name: str = 'distiluse-base-multilingual-cased',
                         model_type: str = 'sentence-transformers',
                         aggregation: str = 'mean',
                         batch_size: int = 32,
                         max_length: int = 512,
                         device: Optional[str] = None) -> pd.DataFrame:
    """
    Encode MSCOCO caption lists and aggregate them.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing MSCOCO caption lists
    mscoco_column : str, default 'mscoco_captions'
        Column name containing lists of MSCOCO captions
    model_name : str, default 'bert-base-multilingual-cased'
        Model name for encoding
    model_type : str, default 'transformers'
        Type of model loading
    aggregation : str, default 'mean'
        How to aggregate multiple captions: 'mean', 'max', 'concat', 'individual'
    batch_size : int, default 32
        Batch size for encoding
    max_length : int, default 512
        Maximum sequence length
    device : str, optional
        Device to use
        
    Returns
    -------
    pd.DataFrame
        DataFrame with MSCOCO embedding column(s)
    """
    df_copy = df.copy()
    
    if mscoco_column not in df.columns:
        logger.error(f"Column '{mscoco_column}' not found in DataFrame")
        return df_copy
    
    # Flatten all MSCOCO captions
    all_captions = []
    caption_indices = []  # Track which row each caption belongs to
    
    for row_idx, caption_list in enumerate(df[mscoco_column]):
        if caption_list is not None and isinstance(caption_list, list):
            for caption in caption_list:
                if caption and str(caption).strip():
                    all_captions.append(str(caption).strip())
                    caption_indices.append(row_idx)
    
    if not all_captions:
        logger.warning("No valid MSCOCO captions found")
        return df_copy
    
    logger.info(f"Encoding {len(all_captions)} MSCOCO captions")
    
    # Encode all captions
    embeddings = encode_captions(
        captions=all_captions,
        model_name=model_name,
        model_type=model_type,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    # Aggregate embeddings by row
    if aggregation == 'individual':
        # Store all individual embeddings
        mscoco_embeddings = [[] for _ in range(len(df))]
        for emb, row_idx in zip(embeddings, caption_indices):
            mscoco_embeddings[row_idx].append(emb)
        df_copy['mscoco_embeddings_individual'] = mscoco_embeddings
        
    else:
        # Aggregate embeddings per row
        aggregated_embeddings = []
        for row_idx in range(len(df)):
            row_embeddings = [emb for emb, idx in zip(embeddings, caption_indices) if idx == row_idx]
            
            if row_embeddings:
                row_embeddings = np.array(row_embeddings)
                if aggregation == 'mean':
                    agg_emb = np.mean(row_embeddings, axis=0)
                elif aggregation == 'max':
                    agg_emb = np.max(row_embeddings, axis=0)
                elif aggregation == 'concat':
                    agg_emb = np.concatenate(row_embeddings)
                else:
                    logger.warning(f"Unknown aggregation method: {aggregation}, using mean")
                    agg_emb = np.mean(row_embeddings, axis=0)
                aggregated_embeddings.append(agg_emb)
            else:
                # No valid captions for this row
                if aggregation == 'concat':
                    # Use zero vector with appropriate size
                    emb_dim = embeddings.shape[1] if len(embeddings) > 0 else 768
                    agg_emb = np.zeros(emb_dim * 5)  # Assume 5 captions max
                else:
                    emb_dim = embeddings.shape[1] if len(embeddings) > 0 else 768
                    agg_emb = np.zeros(emb_dim)
                aggregated_embeddings.append(agg_emb)
        
        df_copy[f'mscoco_embeddings_{aggregation}'] = [emb.tolist() for emb in aggregated_embeddings]
    
    logger.info(f"Added MSCOCO embedding column with {aggregation} aggregation")
    return df_copy


def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of recommended models for different use cases.
    
    Returns
    -------
    dict
        Dictionary of model categories and recommended models
    """
    return {
        'multilingual': [
            'distiluse-base-multilingual-cased',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ],
        'english': [
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'bert-base-uncased'
        ],
        'german': [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'deepset/gbert-base'
        ],
        'fast': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-MiniLM-L3-v2'
        ],
        'high_quality': [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ]
    }