import os
import shutil
import sys


def copy_images(source_dir, destination_dir, file_list):
    """
    Copies images listed in a file from a source directory to a destination directory.

    Parameters:
    - source_dir: The directory where the images are located.
    - destination_dir: The directory where the images will be copied to.
    - file_list: A file containing the list of image file names to copy.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Open the file containing the list of images to copy
    with open(file_list, 'r') as file:
        for line in file:
            # Remove any trailing whitespace or newline characters
            image_name = line.strip()

            # Define the source and destination file paths
            source_file = os.path.join(source_dir, image_name)
            destination_file = os.path.join(destination_dir, image_name)

            # Check if the source file exists before attempting to copy
            if os.path.exists(source_file):
                # Copy the file to the destination directory
                shutil.copy(source_file, destination_file)
            else:
                print(f"File {image_name} not found in source directory.")


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <source_dir> <destination_dir> <file_list>")
        sys.exit(1)

    source_dir = sys.argv[1]
    destination_dir = sys.argv[2]
    file_list = sys.argv[3]

    copy_images(source_dir, destination_dir, file_list)


if __name__ == "__main__":
    """
    python scripts/create_eval_data.py /mnt/disks/datasets/celeba-hq ./eval_data ./data/celebahqvalidation_jpg.txt
    """
    main()
