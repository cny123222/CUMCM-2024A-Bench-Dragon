import csv

header = [[f'{second}s' for second in range(0, 301)]]
header.insert(0, '')

with open('my_result1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
