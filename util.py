import csv
import re

def clean_csv_data_team(filepath):
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

def clean_csv_data_game(filepath):
    """
    Cleans a tab-delimited CSV file of match data to match the desired comma-separated format.
    Each field is stripped of extra spaces, and commas within fields (e.g., 'AM,CM') are preserved.
    Output format example: Rasmus Højlund,9,dk DEN,FW,22-061,70,...

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

def formatJson(json):
    return json  # If data is malformed call this function to format correctly TODO: Implement if needed 

def main():
     #Example usage
    #filepath = "data/manUtd/teamStats.csv"
    '''
    games = ["data/manUtd/csv/games/april_13_Newcastle.csv" , "data/manUtd/games/april1_Nottingham.csv" , "data/manUtd/april6_ManCity.csv" , "data/manUtd/april10_Lyon.csv", "data/manUtd/march_16_Leicester.csv"]
    for game in range(len(games)):
        clean_csv_data(game)
    
    clean_csv_data_game("data/manUtd/csv/games/march13_RealSoc.csv")
    clean_csv_data_game("data/manUtd/csv/games/march9_arsenal.csv")
    clean_csv_data_game("data/manUtd/csv/games/march6_RealSoc.csv")
    clean_csv_data_game("data/manUtd/csv/games/march2_Fulham.csv")
    '''
    clean_csv_data_game("data/manUtd/csv/games/march2_Fulham.csv")
    
if __name__ == "__main__":
    main()