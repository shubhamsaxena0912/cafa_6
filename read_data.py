import json
import os


def read_and_print_top3_entries():
    """
    Reads contents from data/Train/train_sequences.fasta and prints the first 3 entries.
    Each entry starts with a line beginning with '>' and includes all sequence lines until the next '>'.
    """
    file_path = 'data/Train/train_sequences.fasta'
    entry_count = 0
    
    with open(file_path, 'r') as f:
        current_entry = []
        
        for line in f:
            line = line.rstrip()
            
            # Check if this is a new entry (starts with '>')
            if line.startswith('>'):
                # If we have a previous entry, print it
                if current_entry:
                    print('\n'.join(current_entry))
                    print()  # Empty line between entries
                    entry_count += 1
                    
                    # Stop after 3 entries
                    if entry_count >= 3:
                        break
                
                # Start new entry
                current_entry = [line]
            else:
                # Add sequence line to current entry
                if current_entry:
                    current_entry.append(line)
        
        # Print the last entry if we haven't reached 3 yet
        if current_entry and entry_count < 3:
            print('\n'.join(current_entry))


def parse_fasta_and_save_mapping():
    """
    Parses the FASTA file and extracts protein_id -> amino_sequence mapping.
    Format: >sp|<protein_id>|...<ignore>
    Saves the mapping to data/Train/protein_amino_acid_mapping.json
    """
    file_path = 'data/Train/train_sequences.fasta'
    output_path = 'data/Train/protein_amino_acid_mapping.json'
    
    protein_mapping = {}
    current_protein_id = None
    current_sequence = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # Check if this is a new entry (starts with '>sp|')
            if line.startswith('>sp|'):
                # Save previous entry if exists
                if current_protein_id is not None and current_sequence:
                    # Join all sequence lines into a single string
                    protein_mapping[current_protein_id] = ''.join(current_sequence)
                
                # Extract protein_id (between first and second '|')
                parts = line.split('|')
                if len(parts) >= 2:
                    current_protein_id = parts[1]
                    current_sequence = []
                else:
                    current_protein_id = None
                    current_sequence = []
            else:
                # Add sequence line to current entry
                if current_protein_id is not None and line:
                    current_sequence.append(line)
        
        # Save the last entry
        if current_protein_id is not None and current_sequence:
            protein_mapping[current_protein_id] = ''.join(current_sequence)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(protein_mapping, f, indent=2)
    
    print(f"Successfully parsed {len(protein_mapping)} protein entries")
    print(f"Mapping saved to {output_path}")
    
    return protein_mapping


if __name__ == '__main__':
    # Uncomment the function you want to run:
    # read_and_print_top3_entries()
    parse_fasta_and_save_mapping()

