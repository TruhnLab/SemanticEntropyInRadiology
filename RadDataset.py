# Defines a custom PyTorch Dataset for radiologic questions

from torch.utils.data import Dataset
from utilFunctions import readCSV


class RadDataset(Dataset):
    # Loads and provides questions, answers, and metadata
    def __init__(self,csvFile):
        # Initializes the dataset from the specified CSV file
        self.head, *self.data = readCSV(csvFile,";")
    
    def __len__(self):
        # Returns the total number of data samples
        return len(self.data)
    
    def __getitem__(self,idx):
        # Retrieves a single sample by index
        line = self.data[idx] 
        retDict = {
            "question": line[1],
            "answer": line[2],
            "subtopic": line[3],
            "reference": line[4],
            "lastUpdate": line[5],
            }
        return retDict