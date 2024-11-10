# CNN MNIST Training Visualization

This project implements a 4-layer CNN trained on MNIST with real-time training visualization.

## Requirements 

```bash
pip install torch torchvision numpy flask matplotlib tqdm
```
## Project Structure

mnist_cnn/
├── HowTo.md
├── train.py
├── model.py
├── templates/
│ └── index.html
├── static/
│ └── style.css
└── server.py
```

## Running the Project

1. Start the Flask server:

```bash
python server.py
```

2. In a new terminal, start the training:

```bash
python train.py
```

3. Open your browser and navigate to:

```
http://127.0.0.1:5000
``` 

You will see real-time training progress, accuracy and loss curves, and final results on random test images.