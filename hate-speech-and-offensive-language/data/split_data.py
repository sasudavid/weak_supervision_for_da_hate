import csv


'''
generate_split_data: splits a provided dataset into 3 main datasets, containing hateful, offensive and neutral comments respectively.

params:
    combined_data: This is a csv file containing all of the labelled speech comments
    hate_data: This is a csv file containing all of the speech comments labelled as hateful
    offensive_data: This is a csv file containing all of the speech comments labelled as offensive
    neutral_data: This is a csv file containing all of the speech comments that are neither hateful or offensive

returns:
    None
'''

def generate_split_data(combined_data, hate_data, offensive_data, neutral_data):
    hate = open(hate_data,'a')
    offensive = open(offensive_data, 'a')
    neutral = open(neutral_data, 'a')


    with open(combined_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row[5] == '0':
                    hate.write(row[6]+'\n')
                elif row[5] == '1':
                    offensive.write(row[6]+'\n')
                else:
                    neutral.write(row[6]+'\n')
                
                line_count += 1
    
    hate.close()
    offensive.close()
    neutral.close()

    return


