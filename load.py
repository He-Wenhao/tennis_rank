import csv




def get_all_match_data(file_path):
    # Create an empty list to store the data
    data = []

    # Open the CSV file and read data
    with open(file_path, mode='r', encoding='utf-8') as file:
        # Use DictReader to read the CSV into dictionaries
        reader = csv.DictReader(file)
        
        # Iterate over each row and append the dictionary to the data list
        for row in reader:
            for ele in row:
                if ele == 'winner_games'  or ele =='loser_games':
                    row[ele] = eval(row[ele])
            data.append(row)
            #print(row)

    # Now 'data' is a list of dictionaries
    # Each dictionary represents a row in the CSV, with keys from the header
    return data

def get_player_list(all_match_data):
    players = []
    for match in all_match_data:
        players.append(match['winner_name'])
        players.append(match['loser_name'])
    players = list(set(players))
    return players


if __name__ =="__main__":
    all_match_data = get_all_match_data('cleaned_atp_matches_2020.csv')
    print(all_match_data)
    print(get_player_list(all_match_data))