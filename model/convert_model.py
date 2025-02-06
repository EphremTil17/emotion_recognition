import torch

# Load checkpoint
checkpoint_path = "best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Save only the model weights (removes unnecessary metadata)
torch.save(checkpoint['model_state_dict'], "best_model_weights.pth")

print("New weights-only checkpoint saved as 'model_weights.pth'")
