from setuptools import find_packages, setup

setup ( 
    name = 'TextCollectionsForClassificationLibrary', 
    packages = find_packages(), 
    version = '0.1.0', 
    description = 'Library to use a text collection in which the texts are app reviews and the collection have 4 classes', 
    author = 'Marcos P. S. Gôlo',  
    install_requires = ['gdown']
)