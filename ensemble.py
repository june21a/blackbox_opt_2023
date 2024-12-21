import os
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default="./submissions", help="path to train result folder")
    parser.add_argument("--save_path", type=str, default="./ensembles", help="where to save ensemble results")
    
    args = parser.parse_args()
    return args


def ensemble_csvs(target_folder, output_file, id_column):
    """
    Averages the 'y' column of all CSV files in a target folder, row by row, and saves the result to a new CSV.
    
    Args:
        target_folder (str): Path to the folder containing the CSV files.
        output_file (str): Path to save the resulting ensemble CSV.
    """
    # List all CSV files in the target folder
    csv_files = [file for file in os.listdir(target_folder) if file.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the target folder.")
        return

    # Read all CSV files and store 'y' columns in a list
    y_values = []
    id_values = None
    for file in csv_files:
        file_path = os.path.join(target_folder, file)
        df = pd.read_csv(file_path)
        
        if 'y' not in df.columns:
            print(f"'y' column not found in {file}. Skipping...")
            continue
        
        # Collect the ID column if specified
        if id_column and id_column in df.columns:
            if id_values is None:
                id_values = df[id_column]
            else:
                if not id_values.equals(df[id_column]):
                    print(f"ID column '{id_column}' does not match across all files. Skipping {file}.")
                    continue
        
        y_values.append(df['y'])

    if not y_values:
        print("No 'y' columns found in the CSV files.")
        return

    # Concatenate all 'y' columns and calculate the row-wise mean
    y_ensemble = pd.concat(y_values, axis=1).mean(axis=1)

    # Save the resulting ensemble to a new CSV
    output_df = pd.DataFrame({'y': y_ensemble})
    if id_values is not None:
        output_df.insert(0, id_column, id_values)
    output_df.to_csv(output_file, index=False)
    print(f"Ensemble results saved to {output_file}")


def main():
    args = parse_arguments()
    save_file_path = os.path.join(args.save_path, "ensemble.csv")
    os.makedirs(args.save_path, exist_ok=True)
    
    ensemble_csvs(args.target_folder, save_file_path, "ID")

if __name__=="__main__":
    main()