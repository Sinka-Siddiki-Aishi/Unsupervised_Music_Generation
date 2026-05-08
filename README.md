
# Unsupervised Neural Network for Music Generation

## Course Information

**Course:** CSE425 / EEE474 Neural Networks  
**Project Title:** Unsupervised Neural Network for Multi-Genre Music Generation  
**Project Type:** Neural Network Based Symbolic Music Generation  

---

## Project Overview

This project focuses on symbolic music generation using unsupervised and generative neural network models. The main goal is to train models that can learn musical patterns from MIDI files and generate new music samples.

The project implements four major tasks:

1. **Task 1:** LSTM Autoencoder Music Generator  
2. **Task 2:** Variational Autoencoder Music Generator  
3. **Task 3:** Transformer-Based Music Generator  
4. **Task 4:** RLHF / Human Preference Tuning  

Tasks 1, 2, and 3 are mandatory. Task 4 is implemented as an optional bonus task using human listening scores as a reward signal.

---

## Dataset

### Selected Dataset

This project uses the **MAESTRO Dataset**, one of the official MIDI datasets allowed in the project guideline.

A subset of the dataset was used due to computational limitations. Although the dataset may contain both audio and MIDI files, only the MIDI files were used for model training because this project focuses on symbolic music generation.

### Dataset Representation

The MIDI files were converted into piano-roll format.

After preprocessing, the final piano-roll dataset had the shape:

(62084, 128, 128)

This means the dataset contains 62,084 processed music sequences. Each sequence has 128 time steps, and each time step contains 128 MIDI pitch values.


Midi Files
https://drive.google.com/file/d/1393MdFLBmn5DHZZjYTmKUM-yr9Qq2TXf/view?usp=sharing
