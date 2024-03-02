# Code Execution Guide

This guide outlines the steps to run the provided code for Parts-of-Speech (POS) tagging using Feed Forward Neural Network and Recurrent Neural Network models.

## Prerequisites

- Python 3.x
- Virtual environment (virtualenv or venv) dont forget to activate it before running code if using

## Installation

1. **Install Virtual Environment:**
   ```bash
   pip install virtualenv  # if not already installed
    ```
2. **Create and Activate Virtual Environment:**
    ```bash
    virtualenv venv  # Create virtual environment
    source venv/bin/activate  # Activate virtual environment (Unix/Mac)
    # or
    venv\Scripts\activate  # Activate virtual environment (Windows)
    ```
3. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Data and Model Files**
    Ensure the following files are present in the working directory:

    - conllu files: <en_atis-ud-dev.conllu>, <en_atis-ud-train.conllu>, <en_atis-ud-test.conllu>
    - GloVe embedding file: <glove.6B.100d.txt>
    - Model files 
    
        (Everything provided in the link)
    - [Models](https://drive.google.com/drive/folders/1QT-imzPfGdqyroeYIL1hsbTvq4eyZXvV?usp=sharing)
    - [Other files(Embedding.txt, conllu)](https://drive.google.com/drive/folders/1wCnN5HFSivd7L19O8kXHHqXnmDrD-lBL?usp=sharing)

    
5. **Running the Code**
    Feed Forward Neural Network POS Tagging

    - To run the Feed Forward Neural Network POS Tagger, execute the following command:
    ```bash
    python3 pos_tagger.py -f
    or
    python pos_tagger.py -f
    ```
    
    Recurrent Neural Network POS Tagging

    - To run the Recurrent Neural Network POS Tagger, execute the following command:
    ```bash
    python3 pos_tagger.py -r
    or
    python pos_tagger.py -r
    ```

