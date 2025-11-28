# Utility functions for reading from and appending to CSV files

import csv

def readCSV(inpFile,delimiter=","):
    # Reads the CSV file into a list of rows
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def appendLineToCSV(file,lst: list):
    # Appends a new row to the CSV file
    with open(file,"a",newline="") as f: 
        csv.writer(f).writerow(lst)