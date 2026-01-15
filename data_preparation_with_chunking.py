import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_protein_sequences(json_path):
    """
    Load protein ID to amino acid sequence mapping from JSON file.
    
    Args:
        json_path: Path to the JSON file containing protein sequences
        
    Returns:
        dict: Dictionary mapping protein IDs to amino acid sequences
    """
    with open(json_path, 'r') as f:
        protein_sequences = json.load(f)
    print(f"Loaded {len(protein_sequences)} protein sequences from {json_path}")
    return protein_sequences


def load_go_terms(tsv_path):
    """
    Load GO terms from TSV file and group by EntryID.
    
    Args:
        tsv_path: Path to the TSV file containing GO terms
        
    Returns:
        dict: Dictionary mapping EntryID to list of GO terms
    """
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} GO term entries from {tsv_path}")
    
    # Group by EntryID and collect all GO terms
    go_terms_dict = {}
    for entry_id, group in df.groupby('EntryID'):
        go_terms_dict[entry_id] = group['term'].tolist()
    
    print(f"Found {len(go_terms_dict)} unique proteins with GO terms")
    return go_terms_dict


def map_sequences_to_labels(protein_sequences, go_terms_dict):
    """
    Map amino acid sequences to their corresponding GO labels.
    Only includes proteins that have both sequence and GO terms.
    
    Args:
        protein_sequences: Dictionary mapping protein IDs to sequences
        go_terms_dict: Dictionary mapping EntryID to list of GO terms
        
    Returns:
        list: List of tuples (sequence, go_terms_list) for proteins with both
    """
    mapped_data = []
    
    for protein_id, sequence in protein_sequences.items():
        if protein_id in go_terms_dict:
            go_terms = go_terms_dict[protein_id]
            mapped_data.append({
                'protein_id': protein_id,
                'sequence': sequence,
                'go_terms': go_terms
            })
    
    print(f"Mapped {len(mapped_data)} proteins with both sequences and GO terms")
    return mapped_data


def split_data(mapped_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Randomly split data into train, validation, and test sets.
    
    Args:
        mapped_data: List of data dictionaries
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.1)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Convert to numpy array for easier indexing
    data_array = np.array(mapped_data, dtype=object)
    
    # First split: separate train from (val + test)
    train_data, temp_data = train_test_split(
        data_array,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: separate val from test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=random_state,
        shuffle=True
    )
    
    # Convert back to lists
    train_data = train_data.tolist()
    val_data = val_data.tolist()
    test_data = test_data.tolist()
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(mapped_data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/len(mapped_data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(mapped_data)*100:.1f}%)")
    
    return train_data, val_data, test_data


def chunk_sequences(data, max_sequence_len=512, stride=256):
    """
    Break longer sequences into shorter chunks.
    
    Args:
        data: List of data dictionaries with 'sequence' and 'go_terms'
        max_sequence_len: Maximum length of each chunk (default: 512)
        stride: Step size between chunk starts (default: 256)
        
    Returns:
        list: List of chunked data dictionaries
    """
    chunked_data = []
    
    for item in data:
        sequence = item['sequence']
        go_terms = item['go_terms']
        protein_id = item['protein_id']
        seq_len = len(sequence)
        
        # If sequence is shorter than max_sequence_len, keep as is
        if seq_len <= max_sequence_len:
            chunked_data.append({
                'protein_id': protein_id,
                'sequence': sequence,
                'go_terms': go_terms,
                'chunk_index': 0,
                'original_length': seq_len
            })
        else:
            # Break into chunks with stride
            chunk_index = 0
            start = 0
            
            while start < seq_len:
                end = min(start + max_sequence_len, seq_len)
                chunk_sequence = sequence[start:end]
                
                chunked_data.append({
                    'protein_id': protein_id,
                    'sequence': chunk_sequence,
                    'go_terms': go_terms,  # Same labels for all chunks
                    'chunk_index': chunk_index,
                    'original_length': seq_len,
                    'chunk_start': start,
                    'chunk_end': end
                })
                
                chunk_index += 1
                start += stride
                
                # If the last chunk would be very short, we can stop
                # But let's include it anyway as per user's requirement
    
    return chunked_data


def save_data(data, output_path):
    """
    Save data to JSON file.
    
    Args:
        data: List of data dictionaries to save
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    """
    Main function to prepare and preprocess the data.
    """
    # File paths
    protein_mapping_path = 'data/Train/protein_amino_acid_mapping.json'
    train_terms_path = 'data/Train/train_terms.tsv'
    
    # Output paths
    output_dir = 'data/processed'
    # Raw data paths
    train_output_path = os.path.join(output_dir, 'train_data.json')
    val_output_path = os.path.join(output_dir, 'val_data.json')
    test_output_path = os.path.join(output_dir, 'test_data.json')
    # Chunked data paths (will be set in Step 7)
    
    print("=" * 60)
    print("Data Preparation and Preprocessing")
    print("=" * 60)
    
    # Step 1: Load protein sequences
    print("\n[Step 1] Loading protein sequences...")
    protein_sequences = load_protein_sequences(protein_mapping_path)
    
    # Step 2: Load GO terms
    print("\n[Step 2] Loading GO terms...")
    go_terms_dict = load_go_terms(train_terms_path)
    
    # Step 3: Map sequences to GO labels
    print("\n[Step 3] Mapping sequences to GO labels...")
    mapped_data = map_sequences_to_labels(protein_sequences, go_terms_dict)
    
    # Step 4: Split data into train/val/test
    print("\n[Step 4] Splitting data into train/validation/test sets...")
    train_data, val_data, test_data = split_data(
        mapped_data,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_state=42
    )
    
    # Step 5: Save raw data
    print("\n[Step 5] Saving raw data...")
    save_data(train_data, train_output_path)
    save_data(val_data, val_output_path)
    save_data(test_data, test_output_path)
    
    # Step 6: Chunk sequences
    print("\n[Step 6] Chunking sequences...")
    max_sequence_len = 512
    stride = 256
    
    print(f"  Chunking parameters: max_len={max_sequence_len}, stride={stride}")
    
    train_chunks = chunk_sequences(train_data, max_sequence_len=max_sequence_len, stride=stride)
    val_chunks = chunk_sequences(val_data, max_sequence_len=max_sequence_len, stride=stride)
    test_chunks = chunk_sequences(test_data, max_sequence_len=max_sequence_len, stride=stride)
    
    print(f"  Train chunks: {len(train_chunks)} (from {len(train_data)} original sequences)")
    print(f"  Val chunks: {len(val_chunks)} (from {len(val_data)} original sequences)")
    print(f"  Test chunks: {len(test_chunks)} (from {len(test_data)} original sequences)")
    
    # Step 7: Save chunked data
    print("\n[Step 7] Saving chunked data...")
    train_chunks_path = os.path.join(output_dir, 'train_data_chunks.json')
    val_chunks_path = os.path.join(output_dir, 'val_data_chunks.json')
    test_chunks_path = os.path.join(output_dir, 'test_data_chunks.json')
    
    save_data(train_chunks, train_chunks_path)
    save_data(val_chunks, val_chunks_path)
    save_data(test_chunks, test_chunks_path)
    
    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)
    
    # Print some statistics
    print("\nStatistics:")
    print(f"  Total proteins processed: {len(mapped_data)}")
    print(f"  Average GO terms per protein: {np.mean([len(item['go_terms']) for item in mapped_data]):.2f}")
    print(f"  Average sequence length: {np.mean([len(item['sequence']) for item in mapped_data]):.2f}")
    print(f"\nChunking statistics:")
    print(f"  Total chunks created: {len(train_chunks) + len(val_chunks) + len(test_chunks)}")
    print(f"  Average chunk length (train): {np.mean([len(item['sequence']) for item in train_chunks]):.2f}")
    print(f"  Average chunk length (val): {np.mean([len(item['sequence']) for item in val_chunks]):.2f}")
    print(f"  Average chunk length (test): {np.mean([len(item['sequence']) for item in test_chunks]):.2f}")


if __name__ == '__main__':
    main()

