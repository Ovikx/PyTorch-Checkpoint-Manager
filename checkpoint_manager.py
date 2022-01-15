import os
import torch
import time

# TODO: self.purge() isn't deleting the right things because the sorting algorithm isn't helping

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
    
    def save(self):
        dir_contents = os.listdir(self.directory)

        if len(dir_contents) > 0:
            index = max([self.index_from_file(file_name) for file_name in dir_contents]) + 1
        else:
            index = 1

        with open(f'{self.directory}{self.file_name}_{index}.{self.file_format}', 'w') as f:
            print('Saved file!')
        
        self.purge()
        
    def purge(self):
        dir_contents = sorted(os.listdir(self.directory), reverse=True)

        if len(dir_contents) > self.maximum:
            removals = dir_contents[self.maximum:]
            for elem in removals:
                os.remove(f'{self.directory}{elem}')

manager = CheckpointManager({}, 'training_checkpoints', 'model', file_format='txt')
print(manager.save())