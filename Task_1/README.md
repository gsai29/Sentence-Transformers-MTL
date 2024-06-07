# Task 1 - Sentence Transformer

## Overview
This directory contains the Python script `run_task_1.py`, which demonstrates the generation of sentence embeddings using a pre-trained BERT model from the transformers library. Detailed Explanation in Task_1.pdf. 

## Prerequisites
- Docker installed on your machine

## Setup and Execution

### Step 1: Navigate to Task_1

```bash 
cd Sentence_Transformers_MTL/Task_1
```

### Step 2: Build the Docker Image

```bash
docker build -t sentence-transformer .

```

### Step 3: Run the Docker Container

```bash 
docker run sentence-transformer
```