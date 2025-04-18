import csv
import re

def clean_csv_data(filepath):
    """
    Cleans a CSV file to match the desired format with comma-separated fields.
    Each field is stripped of extra spaces, and commas within fields (e.g., 'MF,FW') are preserved.
    Output format example: André Onana,cm,CMR,GK,29-016,31,...

    Args:
        filepath (str): Path to the CSV file to clean
    """
    # Temporary storage for cleaned rows
    cleaned_rows = []
    
    # Read the CSV file
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')  # Assuming tab-delimited input
            for row in reader:
                # Strip whitespace from each field and handle empty fields
                cleaned_row = [field.strip() if field.strip() else '' for field in row]
                cleaned_rows.append(cleaned_row)
    
        # Write the cleaned content back to the file
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for row in cleaned_rows:
                writer.writerow(row)
        
        print(f"Successfully cleaned {filepath}")
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        
def main():
    # Example usage
    filepath = "data/manUtd/teamStats.csv"
    clean_csv_data(filepath)

if __name__ == "__main__":
    main()