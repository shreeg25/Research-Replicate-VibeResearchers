import os

def map_directory(target_path):
    print(f"\n========== SCANNING: {target_path} ==========")
    if not os.path.exists(target_path):
        print(f"CRITICAL ERROR: The folder '{target_path}' DOES NOT EXIST in this directory.")
        return
        
    file_count = 0
    for root, dirs, files in os.walk(target_path):
        for file in files:
            print(os.path.join(root, file))
            file_count += 1
            
    if file_count == 0:
        print("WARNING: This folder is completely empty!")

if __name__ == "__main__":
    print("CURRENT WORKING DIRECTORY:", os.getcwd())
    map_directory("Puthumala-Training_data")
    map_directory("Wayanad_validation_data")