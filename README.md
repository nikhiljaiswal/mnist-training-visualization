# 📊 CNN MNIST Training Visualization
This project demonstrates a 4-layer Convolutional Neural Network (CNN) trained on the MNIST dataset with real-time visualization of training progress, including accuracy and loss tracking. It leverages PyTorch for model building and training, and Flask for an interactive web-based visualization interface.

## ✨ Key Features
- 📈 **Real-time Accuracy and Loss Tracking**: See your model’s performance evolve as it trains.
- 🎯 **Live Updates on Training and Validation Accuracy**: Focused on achieving 90-100% accuracy.
- 🔄 **Interactive Progress Visualization**: View updates using interactive Plotly.js plots.
- ⚙️ **Batch Processing Optimization**: Supports adjustable batch sizes, defaulted to 128.
- 🎲 **Test Results on Random Images**: Observe final predictions on a set of random test images.
- 🌐 **Clean, Responsive Web Interface**: Built with HTML/CSS and JavaScript for a smooth experience.

## 🛠️ Tech Stack
- **Frameworks and Libraries**: PyTorch, Flask, Plotly.js
- **Additional Tools**: `tqdm` for progress bars, `matplotlib` for visualizations
- **Hardware Support**: CUDA-enabled GPU for faster training (optional)

## 📦 Requirements
To install the dependencies, run:
```bash
pip install torch torchvision numpy flask matplotlib tqdm requests
```

## 🚀 Running the Project

**1. Start the Flask server: Open a terminal and run:**

```bash
python server.py
```

**2. Start the training: In a new terminal window, run:**
```bash
python train.py
```

**3. Access the Real-Time Training Dashboard: Open a web browser and go to:**
```bash
http://127.0.0.1:9000
```

  
## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/FeF3EwbDlmA/0.jpg)](https://www.youtube.com/watch?v=FeF3EwbDlmA)
