# in the name of God the most compassionate the most merciful 
# lets create a tree function that returns all the subfolders and files just like os.walk
# we use yield to make it a generator 

import os
import pathlib

def get_subdirectories(directory_path):
    for name in os.listdir(directory_path):
        full_name = os.path.join(directory_path, name)
        if os.path.isdir(full_name):
            yield directory_path, name, full_name

def get_directory(current_path, level=0):
    current_path = (0,0,current_path) if type(current_path) == str else current_path
    yield os.path.split(current_path[-1])[-1], level
    for sub in get_subdirectories(current_path[-1]):
        yield from get_directory(sub, level+1)

def get_dirs(current_path):
    for dir, level in get_directory(current_path):
        print(f'{level * " "*2} {dir}')
        
if __name__ == '__main__':
    get_dirs('/home/hossein/Downloads/')