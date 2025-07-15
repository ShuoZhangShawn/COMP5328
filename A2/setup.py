from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    description="A package for Assignment 2 with CNN models and preprocessing utils.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=['utils', 'utils.*', 'data', 'data.*']),  
    install_requires=[
        'numpy',  # Add your required packages here
        'torch', 
        'matplotlib',  # For visualizations
    ],
    package_data={
        '': ['*.npz'],  # Include dataset files (if needed)
    },
)