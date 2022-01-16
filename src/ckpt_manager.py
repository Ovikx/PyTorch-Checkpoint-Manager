import os
import torch

class CheckpointManager:
    def __init__(self, assets, directory, file_name, maximum=3, file_format='pt'):
        self.assets = assets
        self.directory = directory
        if self.directory[-1] != '/':
            self.directory += '/'
        self.file_name = file_name
        self.maximum = maximum
        self.file_format = file_format
    
    def index_from_file(self, file_name):
        return int(file_name[len(self.file_name)+1:-(len(self.file_format)+1)])
    
    def save(self, epoch=None):
        dir_contents = os.listdir(self.directory)

        if len(dir_contents) > 0 and epoch == None:
            index = max([self.index_from_file(file_name) for file_name in dir_contents]) + 1
        else:
            index = epoch if epoch != None else 1
        
        save_dir = f'{self.directory}{self.file_name}_{index}.{self.file_format}'
        torch.save(self.assets, save_dir)
        
        self.purge()
        print(f'Saved states to {save_dir}')
        
    def purge(self):
        dir_contents = os.listdir(self.directory)
    
        if len(dir_contents) > self.maximum:
            indices = sorted([self.index_from_file(v) for v in dir_contents])
            removals = []
            for index in indices[:len(indices)-self.maximum]:
                for directory in dir_contents:
                    if f'{index}.{self.file_format}' in directory:
                        removals.append(directory)

            for elem in removals:
                os.remove(f'{self.directory}{elem}')
    
    def load(self):
        dir_contents = os.listdir(self.directory)

        for directory in dir_contents:
            if self.file_name in directory:
                max_index = max([self.index_from_file(v) for v in dir_contents])
                load_dir = f'{self.directory}{self.file_name}_{max_index}.{self.file_format}'
                print(f'Loading states from {load_dir}')
                return torch.load(load_dir)
        
        print('Initializing fresh states')
        return self.assets