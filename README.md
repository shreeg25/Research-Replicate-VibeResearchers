# Landslide EEGMoE Replication

This project implements a simplified replication of the "Efficient and Effective Gating for Mixture-of-Experts" (EEGMoE) architecture, adapted for the critical task of geospatial landslide prediction. It focuses on demonstrating cross-domain generalization by training a model on multi-modal satellite data from one geographical region (Puthumala, India) and evaluating its zero-shot performance on an entirely unseen region (Wayanad, India). The model leverages a domain-decoupled Mixture-of-Experts approach to handle diverse geospatial features, aiming for robust performance in new, unobserved areas.

## Feature Set

*   **Multi-modal Geospatial Data Handling**: Processes diverse satellite data types, including Synthetic Aperture Radar (SAR) TIFFs, Optical (Sentinel-2) TIFFs, Soil Moisture TIFFs, and Rainfall NetCDF files.
*   **Robust Data Preprocessing**: Includes utilities for reading, cleaning (NaNs, extreme values), and standardizing geospatial raster data to a uniform `64x64` resolution.
*   **EEGMoE Architecture Implementation**: Features a `Geospatial_To_MoE_Encoder` to transform 2D multi-channel rasters into a 1D sequence, followed by a `SSMoE_Block` (Specific-Shared Mixture of Experts) for domain-decoupled learning.
*   **Cross-Domain Generalization**: Designed for training on a source domain (Puthumala) and evaluating zero-shot on a distinct target domain (Wayanad) to assess transferability.
*   **Load-Balancing Auxiliary Loss**: Incorporates the `L_aux` loss term to encourage balanced expert utilization within the specific MoE.
*   **Routing Distribution Visualization**: Generates a bar chart illustrating how geographical tokens are routed to different experts, providing insights into the model's decision-making.
*   **Data Directory Validation**: A utility script (`check_data.py`) to verify the presence and structure of required input data folders.

## Project Structure

```
landslide_eegmoe_replication/
├── .vscode/
│   └── settings.json
├── Puthumala-Training_data/
│   └── ... (Contains multi-modal satellite data for training)
├── Wayanad_validation_data/
│   └── ... (Contains multi-modal satellite data for zero-shot validation)
├── check_data.py           # Utility to verify data directory structure.
├── data_loader.py          # Defines data loading and preprocessing for multi-modal satellite imagery.
├── models.py               # Implements the EEGMoE architecture components.
├── plot_routing.py         # Generates a visualization of expert routing distribution.
└── train.py                # Main script for training and cross-domain evaluation.
└── usp_explainability.py   #Script for peeking inside the MoE block during validation.
```

## Technical Stack

*   **Python**: Core programming language.
*   **PyTorch**: Deep learning framework for model definition, training, and evaluation.
*   **NumPy**: Fundamental package for numerical computing.
*   **Rasterio**: For reading and writing geospatial raster datasets (TIFF).
*   **Xarray**: For working with labeled multi-dimensional arrays, particularly NetCDF files.
*   **Matplotlib**: For creating static, interactive, and animated visualizations (e.g., routing plot).

## Setup

Follow these steps to get the project up and running:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/landslide_eegmoe_replication.git
    cd landslide_eegmoe_replication
    ```

2.  **Create and Activate a Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Install the required Python packages.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or appropriate CUDA/CPU version
    pip install numpy rasterio xarray matplotlib
    ```
    *(Note: The `torch` installation command above is for CUDA 11.8. Adjust `--index-url` if you have a different CUDA version or want a CPU-only build.)*

4.  **Download and Place Data**:
    The project expects two main data directories: `Puthumala-Training_data` and `Wayanad_validation_data`. Each directory should contain the multi-modal satellite data (SAR.tif, B04.tif, SM_SMAP_*.tif, *.nc) as described in `data_loader.py`.
    *   **Crucially**: Ensure these folders are placed directly inside the `landslide_eegmoe_replication` directory.
    *   *Placeholder for data download instructions*: (If data is hosted online, provide links and instructions here. Otherwise, users are expected to provide their own data matching the expected structure.)

5.  **Verify Data Structure**:
    Run the `check_data.py` script to ensure your data directories are correctly set up and contain the expected files.
    ```bash
    python check_data.py
    ```
    This will scan the `Puthumala-Training_data` and `Wayanad_validation_data` folders and report any missing files or empty directories.

6.  **Train and Evaluate the Model**:
    Execute the main training script. This will train the model on the Puthumala data and then perform a zero-shot evaluation on the Wayanad data.
    ```bash
    python train.py
    ```
    The final cross-domain accuracy will be printed to the console.

7.  **Visualize Routing Distribution**:
    After training (or even independently, as it uses hardcoded data for demonstration), you can generate a plot showing the expert routing.
    ```bash
    python plot_routing.py
    ```
    This will save a `routing_distribution.png` file in the project root.
