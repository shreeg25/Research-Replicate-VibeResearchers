import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loader import HackathonMultiModalDataset
from models import Landslide_EEGMoE

def train_and_evaluate():
    # --- 1. CROSS-DOMAIN CONFIGURATION ---
    TRAIN_DIR = "Puthumala-Training_data"  # Source Domain (2019)
    TEST_DIR = "Wayanad_validation_data"   # Target Domain (2024)
    
    EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4  
    ALPHA = 1e-4          

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Cross-Domain Validation on: {device}")

    # --- 2. INITIALIZATION ---
    print("\nLoading Source Domain (Training) - Puthumala 2019...")
    train_dataset = HackathonMultiModalDataset(TRAIN_DIR, num_samples=64) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\nLoading Target Domain (Testing) - Wayanad 2024...")
    test_dataset = HackathonMultiModalDataset(TEST_DIR, num_samples=32) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Landslide_EEGMoE(in_channels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() 

    # --- 3. TRAINING LOOP (PUTHUMALA) ---
    print("\n--- Phase 1: Training on Source Domain ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions, l_aux = model(inputs)
            loss = criterion(predictions, targets) + (ALPHA * l_aux)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {total_loss/len(train_loader):.4f} | L_aux: {l_aux.item():.4f}")

    # --- 4. ZERO-SHOT EVALUATION LOOP (WAYANAD) ---
    print("\n--- Phase 2: Evaluating on Unseen Target Domain ---")
    model.eval() # Freeze dropout and batchnorm
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions, _ = model(inputs) 
            predictions = predictions.view(-1)
            targets = targets.view(-1)
            predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
            correct_predictions += (predicted_classes == targets).sum().item()
            total_samples += targets.size(0)

    cross_domain_acc = (correct_predictions / total_samples) * 100
    print(f">>> FINAL CROSS-DOMAIN ACCURACY (Wayanad): {cross_domain_acc:.2f}% <<<")

if __name__ == "__main__":
    train_and_evaluate()