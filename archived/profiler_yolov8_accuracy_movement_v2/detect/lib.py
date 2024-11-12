import shutil
import os

def move_folder(input_folder_path, output_folder_path, new_foldername):
    try:
        # Ensure input folder exists
        if not os.path.exists(input_folder_path):
            print(f"Input folder '{input_folder_path}' does not exist.")
            return

        # Ensure output folder exists, create it if it doesn't
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Get the name of the folder to be moved
        folder_name = os.path.basename(input_folder_path)

        # Construct the new path for the folder in the output directory
        new_folder_path = os.path.join(output_folder_path, new_foldername)

        # Move the folder
        shutil.move(input_folder_path, new_folder_path)

        print(f"Folder '{folder_name}' moved successfully from '{input_folder_path}' to '{new_folder_path}' with new name '{new_foldername}'.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
def delete_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove the folder and its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' deleted successfully.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
            
    except Exception as e:
        print(f"An error occurred: {e}")