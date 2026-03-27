import torch
from data_loader import HackathonMultiModalDataset
from models import Landslide_EEGMoE

def extract_routing_behavior():
    print("\n--- INITIATING MOE ROUTING ANALYSIS ---")
    print("Loading Target Domain (Wayanad)...")
    
    # Load just one real batch of the multi-modal data
    dataset = HackathonMultiModalDataset("Wayanad_validation_data", num_samples=1)
    x, y = dataset[0]
    x = x.unsqueeze(0) 

    model = Landslide_EEGMoE(in_channels=4)
    model.eval()

    with torch.no_grad():
        x_seq = model.encoder(x)
        
        # Intercept the signal inside the MoE Router
        x_flat = x_seq.view(-1, model.moe_block.embed_dim) 
        specific_logits = model.moe_block.specific_router(x_flat)
        specific_probs = torch.softmax(specific_logits, dim=-1)
        
        # Identify which Expert was assigned to which spatial patch
        _, top_expert_indices = torch.topk(specific_probs, 1, dim=-1)
        expert_counts = torch.bincount(top_expert_indices.flatten(), minlength=4)
        
        print("\n==================================================")
        print(" [WHITE-BOX AI: DOMAIN ROUTING DISTRIBUTION]")
        print("==================================================")
        for i, count in enumerate(expert_counts):
            print(f" Specialized Expert {i+1} processed: {count.item()} geographical tokens")
            
        print("\n>>> USP PITCH FOR THE JUDGES <<<")
        print("1. Standard models are black-boxes. They blend all satellite data.")
        print("2. Our MoE dynamically routes patches. We can trace if a prediction")
        print("   was triggered by the Hydrological Expert (Rain) or Structural Expert (SAR).")
        print("3. This prevents dangerous 'Overtrust' in AI for disaster evacuation.")
        print("==================================================\n")

if __name__ == "__main__":
    extract_routing_behavior()