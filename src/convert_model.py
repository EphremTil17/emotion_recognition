import torch
import tensorflow as tf
import tensorflowjs as tfjs
from model import EmotionRecognitionModel
import os

def convert_model():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'model')
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(model_dir, 'web'), exist_ok=True)
    
    print("Loading PyTorch model...")
    # Load PyTorch model
    model_path = os.path.join(model_dir, 'best_model.pth')
    model = EmotionRecognitionModel.load_from_checkpoint(model_path, num_classes=5)
    model.eval()
    
    print("Converting to ONNX...")
    # Convert to ONNX first
    onnx_path = os.path.join(model_dir, 'web', 'emotion_model.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, 
                     dummy_input, 
                     onnx_path,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    print("Converting ONNX to TensorFlow...")
    # Convert ONNX to TensorFlow
    import tf2onnx
    import onnx
    
    onnx_model = onnx.load(onnx_path)
    tf_rep = tf2onnx.convert.from_onnx(onnx_model)
    tf_path = os.path.join(model_dir, 'web', 'emotion_model_tf')
    tf_rep.export_graph(tf_path)
    
    print("Converting to TensorFlow.js...")
    # Convert to TensorFlow.js format
    tfjs_path = os.path.join(model_dir, 'web', 'emotion_model_tfjs')
    tfjs.converters.convert_tf_saved_model(
        tf_path,
        tfjs_path
    )
    
    print(f"Conversion complete! Model saved in: {tfjs_path}")
    print("You can now use this model in the frontend.")

if __name__ == '__main__':
    convert_model()