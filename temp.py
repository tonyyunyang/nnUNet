import os
print(f"nnUNet_raw: {os.environ.get('nnUNet_raw', 'Not set')}")
print(f"nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed', 'Not set')}")
print(f"nnUNet_results: {os.environ.get('nnUNet_results', 'Not set')}")