# CS4337 Final Project

This repository contains the complete source code for the CS4337 Final Project.  
All Python scripts used for the project are included and organized for easy execution.  
A `requirements.txt` file is provided listing every library needed to run the code.

---

## Source Code Repository
Scroll to top of page and locate the green "code" button click it and copy the given HTTPS URL

---

## Setup Instructions

Follow these steps to install everything required to run the project:

1. **Download database and insert into /dataset**
    ```bash
    https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset


2. **Clone the repository**
   ```bash
   git clone https://github.com/Da125673/CS4337FinalPoject.git
   cd <CS4337FinalPoject>
   pip install -r requirements.txt


3. **training one-stagecode and running**
    ```bash
    python one-stagecode/train_single_stage.py

    Once training complete run live demo by: 

    python one-stagecode/realtime_single_stage.py


4. **training two-stagecode and running**
    ```bash
    python two-stagecode/CNN_training.py

    Once training complete run live demo by: 

    python two-stagecode/Live_camara.py

