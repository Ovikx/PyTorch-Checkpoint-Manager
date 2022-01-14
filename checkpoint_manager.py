import os

class CheckpointManager:
    def __init__(self, assets, path, maximum):
        self.assets = assets
        self.path = path
        self.maximum = maximum