def read_and_print_top5():
    """
    Reads contents from data/Train/train_sequences.fasta and prints the top 5 rows.
    """
    file_path = 'data/Train/train_sequences.fasta'
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(line.rstrip())
            else:
                break

if __name__ == '__main__':
    read_and_print_top5()

