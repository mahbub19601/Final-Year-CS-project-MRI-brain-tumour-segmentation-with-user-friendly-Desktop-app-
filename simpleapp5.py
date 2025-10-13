import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import nibabel as nib
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tkinterdnd2 import DND_FILES, TkinterDnD
from sklearn.preprocessing import MinMaxScaler

# ---------------- Constants Matching Your Cropping ---------------- #
# We'll fix these to produce 128×128×128 volumes:
X_START, X_END = 56, 184   # 128 in X dimension
Y_START, Y_END = 56, 184   # 128 in Y dimension
Z_START, Z_END = 13, 141   # 128 in Z dimension

# ---------------- Utility Functions ---------------- #
def browse_directory(entry):
    """Let the user pick a directory, insert the path into the Tkinter Entry."""
    directory = filedialog.askdirectory()
    if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

def browse_file(entry):
    """Let the user pick a file, insert the path into the Tkinter Entry."""
    file = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5"), ("All files", "*.*")])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

def drop_event_handler(event, entry):
    """Handle drag-and-drop events; insert the path into the given Entry."""
    data = event.data.strip('{}')
    if os.path.isdir(data) or os.path.isfile(data):
        entry.delete(0, tk.END)
        entry.insert(0, data)
    else:
        messagebox.showerror("Invalid Drop", "Please drop a valid file or folder.")

def find_file_by_keyword(folder, keyword):
    """
    Search for the first .nii file in 'folder' containing 'keyword' in its filename.
    Returns the file path or None if not found.
    """
    nii_files = glob.glob(os.path.join(folder, '*.nii'))
    for f in nii_files:
        if keyword.lower() in os.path.basename(f).lower():
            return f
    return None

# ---------------- Preprocessing Function ---------------- #
def run_preprocessing():
    """
    1) Finds the NIfTI files in the chosen dataset folder using the user-specified keywords.
    2) Normalizes them with MinMaxScaler.
    3) Reassigns label 4 → 3 in the mask.
    4) Crops them to [56:184, 56:184, 13:141].
    5) Saves processed_image.npy and processed_mask.npy in the same folder.
    6) Optionally visualizes a random slice.
    """
    dataset_folder = prep_dataset_entry.get().strip()
    flair_keyword = prep_flair_keyword_entry.get().strip()
    channel2_keyword = prep_channel2_keyword_entry.get().strip()
    channel3_keyword = prep_channel3_keyword_entry.get().strip()
    seg_keyword = prep_seg_keyword_entry.get().strip()
    
    # Basic validation
    if not os.path.isdir(dataset_folder):
        messagebox.showerror("Error", "Dataset folder not found.")
        return
    
    # Initialize MinMaxScaler for normalization
    scaler = MinMaxScaler()
    
    # Locate NIfTI files using your provided keywords
    flair_file = find_file_by_keyword(dataset_folder, flair_keyword)
    channel2_file = find_file_by_keyword(dataset_folder, channel2_keyword)
    channel3_file = find_file_by_keyword(dataset_folder, channel3_keyword)
    seg_file = find_file_by_keyword(dataset_folder, seg_keyword)
    
    if not all([flair_file, channel2_file, channel3_file, seg_file]):
        messagebox.showerror("File Not Found",
                             "One or more modality files (flair, t1ce/t2, seg) not found with the given keywords.")
        return
    
    # Load and normalize each modality
    flair_data = nib.load(flair_file).get_fdata()
    flair_data = scaler.fit_transform(flair_data.reshape(-1, 1)).reshape(flair_data.shape)
    
    channel2_data = nib.load(channel2_file).get_fdata()
    channel2_data = scaler.fit_transform(channel2_data.reshape(-1, 1)).reshape(channel2_data.shape)
    
    channel3_data = nib.load(channel3_file).get_fdata()
    channel3_data = scaler.fit_transform(channel3_data.reshape(-1, 1)).reshape(channel3_data.shape)
    
    # Load and process the segmentation
    seg_data = nib.load(seg_file).get_fdata().astype(np.uint8)
    seg_data[seg_data == 4] = 3  # Reassign label 4 -> 3
    
    # Stack them into [H, W, D, 3]
    combined = np.stack([flair_data, channel2_data, channel3_data], axis=3)
    
    # Crop to [56:184, 56:184, 13:141] => 128×128×128
    combined = combined[X_START:X_END, Y_START:Y_END, Z_START:Z_END]
    seg_data = seg_data[X_START:X_END, Y_START:Y_END, Z_START:Z_END]
    
    # Visualize a random slice if user requested
    visualize = prep_visualize_var.get()
    if visualize and seg_data.shape[2] > 0:
        rand_slice = random.randint(0, seg_data.shape[2] - 1)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(241)
        plt.imshow(combined[:, :, rand_slice, 0], cmap='gray')
        plt.title(f"{flair_keyword} (Ch 0)")
        plt.axis('off')
        
        plt.subplot(242)
        plt.imshow(combined[:, :, rand_slice, 1], cmap='gray')
        plt.title(f"{channel2_keyword} (Ch 1)")
        plt.axis('off')
        
        plt.subplot(243)
        plt.imshow(combined[:, :, rand_slice, 2], cmap='gray')
        plt.title(f"{channel3_keyword} (Ch 2)")
        plt.axis('off')
        
        plt.subplot(244)
        plt.imshow(seg_data[:, :, rand_slice], cmap='jet')
        plt.title("Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Check label distribution
    vals, counts = np.unique(seg_data, return_counts=True)
    try:
        min_label_fraction = float(prep_min_label_entry.get())
    except ValueError:
        min_label_fraction = 0.01
    
    # If there's enough labeled area, we one-hot encode and save
    if len(counts) > 1 and (1 - (counts[0] / counts.sum())) > min_label_fraction:
        seg_encoded = to_categorical(seg_data, num_classes=4)
        
        # Save to .npy in the same folder
        np.save(os.path.join(dataset_folder, "processed_image.npy"), combined)
        np.save(os.path.join(dataset_folder, "processed_mask.npy"), seg_encoded)
        
        messagebox.showinfo("Success", f"Preprocessing completed.\nFiles saved in: {dataset_folder}")
    else:
        messagebox.showinfo("Skipped", "Insufficient labeled volume in mask. Processing skipped.")

# ---------------- Segmentation Inference Function ---------------- #
def run_segmentation():
    """
    1) Loads processed_image.npy (and processed_mask.npy if you want to compare) from the folder.
    2) Loads your trained .h5 model.
    3) Runs inference, producing a predicted segmentation.
    4) Visualizes a random slice from channel 0 (FLAIR) and the predicted mask.
    """
    processed_folder = seg_processed_folder_entry.get().strip()
    model_file = seg_model_entry.get().strip()
    
    # Basic validation
    if not os.path.isdir(processed_folder):
        messagebox.showerror("Error", "Processed data folder not found.")
        return
    if not os.path.isfile(model_file):
        messagebox.showerror("Error", "Model file not found.")
        return
    
    # Attempt to load the preprocessed data
    image_file = os.path.join(processed_folder, "processed_image.npy")
    if not os.path.exists(image_file):
        messagebox.showerror("Error", f"No 'processed_image.npy' found in:\n{processed_folder}")
        return
    
    # Load data
    img = np.load(image_file)  # Shape: [128, 128, 128, 3]
    
    # Optionally load mask if you want to compare
    mask_file = os.path.join(processed_folder, "processed_mask.npy")
    mask_data = None
    if os.path.exists(mask_file):
        mask_data = np.load(mask_file)
        if mask_data.ndim == 4:
            # Convert one-hot back to integer labels
            mask_data = np.argmax(mask_data, axis=3)
    
    # Load trained model
    try:
        model = load_model(model_file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return
    
    # Inference
    # Model expects shape [batch, X, Y, Z, channels]
    pred = model.predict(np.expand_dims(img, axis=0))  # Output shape [1, X, Y, Z, classes]
    pred = np.squeeze(pred, axis=0)                   # [X, Y, Z, classes]
    seg_pred = np.argmax(pred, axis=3)                # [X, Y, Z]
    
    # Visualize a random slice
    if seg_pred.shape[2] > 0:
        slice_idx = random.randint(0, seg_pred.shape[2] - 1)
        
        plt.figure(figsize=(18, 6))
        
        # Input image (channel 0, FLAIR)
        plt.subplot(1, 3, 1)
        plt.imshow(img[:, :, slice_idx, 0], cmap='gray')
        plt.title("Input (FLAIR Channel 0)")
        plt.axis('off')
        
        # If ground truth mask is available
        plt.subplot(1, 3, 2)
        if mask_data is not None:
            plt.imshow(mask_data[:, :, slice_idx], cmap='jet')
            plt.title("Ground Truth Mask")
        else:
            plt.text(0.5, 0.5, "No GT Mask", ha='center', va='center', fontsize=14)
            plt.title("Ground Truth")
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(seg_pred[:, :, slice_idx], cmap='jet')
        plt.title("Predicted Segmentation")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    messagebox.showinfo("Success", "Segmentation completed and displayed!")

# ---------------- Main GUI Setup ---------------- #
def main():
    root = TkinterDnD.Tk()
    root.title("MRI Segmentation Application")
    
    # Create a Notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both')
    
    # Preprocessing tab
    prep_frame = ttk.Frame(notebook, padding=10)
    notebook.add(prep_frame, text="Preprocessing")
    
    # Row 0: dataset folder
    ttk.Label(prep_frame, text="Dataset Folder:").grid(row=0, column=0, sticky=tk.W)
    global prep_dataset_entry
    prep_dataset_entry = ttk.Entry(prep_frame, width=50)
    prep_dataset_entry.grid(row=0, column=1, padx=5)
    ttk.Button(prep_frame, text="Browse", command=lambda: browse_directory(prep_dataset_entry)).grid(row=0, column=2)
    
    prep_dataset_entry.drop_target_register(DND_FILES)
    prep_dataset_entry.dnd_bind('<<Drop>>', lambda e: drop_event_handler(e, prep_dataset_entry))
    
    # Row 1-4: keywords
    ttk.Label(prep_frame, text="FLAIR Keyword:").grid(row=1, column=0, sticky=tk.W)
    global prep_flair_keyword_entry
    prep_flair_keyword_entry = ttk.Entry(prep_frame, width=20)
    prep_flair_keyword_entry.insert(0, "flair")
    prep_flair_keyword_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
    
    ttk.Label(prep_frame, text="Channel2 Keyword (e.g., t1ce/t2):").grid(row=2, column=0, sticky=tk.W)
    global prep_channel2_keyword_entry
    prep_channel2_keyword_entry = ttk.Entry(prep_frame, width=20)
    prep_channel2_keyword_entry.insert(0, "t1ce")
    prep_channel2_keyword_entry.grid(row=2, column=1, sticky=tk.W, padx=5)
    
    ttk.Label(prep_frame, text="Channel3 Keyword (e.g., t1/t2):").grid(row=3, column=0, sticky=tk.W)
    global prep_channel3_keyword_entry
    prep_channel3_keyword_entry = ttk.Entry(prep_frame, width=20)
    prep_channel3_keyword_entry.insert(0, "t1")
    prep_channel3_keyword_entry.grid(row=3, column=1, sticky=tk.W, padx=5)
    
    ttk.Label(prep_frame, text="Segmentation Keyword:").grid(row=4, column=0, sticky=tk.W)
    global prep_seg_keyword_entry
    prep_seg_keyword_entry = ttk.Entry(prep_frame, width=20)
    prep_seg_keyword_entry.insert(0, "seg")
    prep_seg_keyword_entry.grid(row=4, column=1, sticky=tk.W, padx=5)
    
    # Row 5-8: cropping info
    ttk.Label(prep_frame, text="Cropping Parameters:", font=("TkDefaultFont", 10, "bold")).grid(
        row=5, column=0, columnspan=2, pady=(10, 0), sticky=tk.W
    )
    
    ttk.Label(prep_frame, text="x_start:").grid(row=6, column=0, sticky=tk.W)
    global prep_x_start_entry
    prep_x_start_entry = ttk.Entry(prep_frame, width=10)
    prep_x_start_entry.insert(0, "56")  # default from your code
    prep_x_start_entry.grid(row=6, column=1, sticky=tk.W)
    
    ttk.Label(prep_frame, text="x_end:").grid(row=6, column=2, sticky=tk.W)
    global prep_x_end_entry
    prep_x_end_entry = ttk.Entry(prep_frame, width=10)
    prep_x_end_entry.insert(0, "184")  # default from your code
    prep_x_end_entry.grid(row=6, column=3, sticky=tk.W, padx=(5, 0))
    
    ttk.Label(prep_frame, text="y_start:").grid(row=7, column=0, sticky=tk.W)
    global prep_y_start_entry
    prep_y_start_entry = ttk.Entry(prep_frame, width=10)
    prep_y_start_entry.insert(0, "56")
    prep_y_start_entry.grid(row=7, column=1, sticky=tk.W)
    
    ttk.Label(prep_frame, text="y_end:").grid(row=7, column=2, sticky=tk.W)
    global prep_y_end_entry
    prep_y_end_entry = ttk.Entry(prep_frame, width=10)
    prep_y_end_entry.insert(0, "184")
    prep_y_end_entry.grid(row=7, column=3, sticky=tk.W, padx=(5, 0))
    
    ttk.Label(prep_frame, text="z_start:").grid(row=8, column=0, sticky=tk.W)
    global prep_z_start_entry
    prep_z_start_entry = ttk.Entry(prep_frame, width=10)
    prep_z_start_entry.insert(0, "13")
    prep_z_start_entry.grid(row=8, column=1, sticky=tk.W)
    
    ttk.Label(prep_frame, text="z_end:").grid(row=8, column=2, sticky=tk.W)
    global prep_z_end_entry
    prep_z_end_entry = ttk.Entry(prep_frame, width=10)
    prep_z_end_entry.insert(0, "141")
    prep_z_end_entry.grid(row=8, column=3, sticky=tk.W, padx=(5, 0))
    
    # Min label fraction
    ttk.Label(prep_frame, text="Min Label Fraction:").grid(row=9, column=0, sticky=tk.W, pady=(10, 0))
    global prep_min_label_entry
    prep_min_label_entry = ttk.Entry(prep_frame, width=10)
    prep_min_label_entry.insert(0, "0.01")
    prep_min_label_entry.grid(row=9, column=1, sticky=tk.W, pady=(10, 0))
    
    # Visualization checkbox
    global prep_visualize_var
    prep_visualize_var = tk.BooleanVar()
    prep_visualize_check = ttk.Checkbutton(prep_frame, text="Visualize Random Slice", variable=prep_visualize_var)
    prep_visualize_check.grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    # Preprocessing button
    ttk.Button(prep_frame, text="Run Preprocessing", command=run_preprocessing).grid(
        row=11, column=0, columnspan=4, pady=(15, 0)
    )
    
    # Segmentation Inference tab
    seg_frame = ttk.Frame(notebook, padding=10)
    notebook.add(seg_frame, text="Segmentation Inference")
    
    ttk.Label(seg_frame, text="Processed Data Folder:").grid(row=0, column=0, sticky=tk.W)
    global seg_processed_folder_entry
    seg_processed_folder_entry = ttk.Entry(seg_frame, width=50)
    seg_processed_folder_entry.grid(row=0, column=1, padx=5)
    ttk.Button(seg_frame, text="Browse", command=lambda: browse_directory(seg_processed_folder_entry)).grid(row=0, column=2)
    seg_processed_folder_entry.drop_target_register(DND_FILES)
    seg_processed_folder_entry.dnd_bind('<<Drop>>', lambda e: drop_event_handler(e, seg_processed_folder_entry))
    
    ttk.Label(seg_frame, text="Trained Model File (.h5):").grid(row=1, column=0, sticky=tk.W)
    global seg_model_entry
    seg_model_entry = ttk.Entry(seg_frame, width=50)
    seg_model_entry.grid(row=1, column=1, padx=5)
    ttk.Button(seg_frame, text="Browse", command=lambda: browse_file(seg_model_entry)).grid(row=1, column=2)
    seg_model_entry.drop_target_register(DND_FILES)
    seg_model_entry.dnd_bind('<<Drop>>', lambda e: drop_event_handler(e, seg_model_entry))
    
    ttk.Button(seg_frame, text="Run Segmentation", command=run_segmentation).grid(row=2, column=0, columnspan=3, pady=(15, 0))
    
    root.mainloop()

if __name__ == "__main__":
    main()