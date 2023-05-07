import csv
import sys

def split_csv(input_file, output_file1, output_file2):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        with open(output_file1, 'w') as out1, open(output_file2, 'w') as out2:
            writer1 = csv.writer(out1)
            writer2 = csv.writer(out2)

            writer1.writerow(headers)
            writer2.writerow(headers)

            for i, row in enumerate(reader):
                if i % 2 == 0:
                    writer1.writerow(row)
                else:
                    writer2.writerow(row)

def join_csv(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as csvfile1, open(input_file2, 'r') as csvfile2:
        reader1 = csv.reader(csvfile1)
        reader2 = csv.reader(csvfile2)

        headers1 = next(reader1)
        headers2 = next(reader2)

        if headers1 != headers2:
            print("Error: CSV files have different headers")
            sys.exit(1)

        with open(output_file, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers1)

            rows1 = list(reader1)
            rows2 = list(reader2)

            for i in range(max(len(rows1), len(rows2))):
                if i < len(rows1):
                    writer.writerow(rows1[i])
                if i < len(rows2):
                    writer.writerow(rows2[i])

if __name__ == '__main__':
    operation = input("Enter 'split' to split a CSV file or 'join' to join two CSV files: ").lower()

    if operation == 'split':
        input_file = input("Enter the input CSV file path: ")
        output_file1 = input("Enter the first output CSV file path: ")
        output_file2 = input("Enter the second output CSV file path: ")
        split_csv(input_file, output_file1, output_file2)
    elif operation == 'join':
        input_file1 = input("Enter the first input CSV file path: ")
        input_file2 = input("Enter the second input CSV file path: ")
        output_file = input("Enter the output CSV file path: ")
        join_csv(input_file1, input_file2, output_file)
    else:
        print("Invalid operation. Please enter 'split' or 'join'.")
