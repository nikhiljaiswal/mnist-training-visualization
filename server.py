from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Global variables to store training data
losses = []
train_accuracies = []
epochs = []
final_results = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    data = request.json
    
    # Check if we already have this epoch-batch combination
    current_epoch = data['epoch']
    
    # Update the latest values
    if len(epochs) > 0 and epochs[-1] == current_epoch:
        losses[-1] = data['loss']
        train_accuracies[-1] = data['train_acc']
    else:
        losses.append(data['loss'])
        train_accuracies.append(data['train_acc'])
        epochs.append(current_epoch)
    
    return jsonify({'status': 'success'})

@app.route('/get_data')
def get_data():
    return jsonify({
        'losses': losses,
        'accuracies': train_accuracies,
        'epochs': epochs
    })

@app.route('/final_results', methods=['POST'])
def save_final_results():
    global final_results
    final_results = request.json
    return jsonify({'status': 'success'})

@app.route('/get_final_results')
def get_final_results():
    return jsonify(final_results if final_results else {})

if __name__ == '__main__':
    app.run(debug=True, port=9000) 