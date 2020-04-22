import setuptools

with open("README.md", "r") as ld:
    long_description = ld.read()

setuptools.setup(
    name = "relation-detection-LSTM",
    version = "20.01",
    author = "Cyrielle Mallart",
    author_email = "cyrielle.mallart@ouest-france.fr",
    description = "Code base for the paper \"Efficient LSTM-based relation detection to improve knowledge extraction\"",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    python_requires = ">=3",
    install_requires =['keras', 
                    'numpy', 
                    'matplotlib', 
                    'gensim', 
                    'pandas',
                    'time',
                    'click', 
                    'collections', 
                    'warnings',
                    'nltk',
                    'tqdm',
                    'networkx',
                    're',
                    'json',
                    'sqlite3',
                    'bz2',
                    'urllib',
                    'PyStemmer'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={'relationdetection': ['data/*.db']}
)

#TODO : add url
#TODO : the readme