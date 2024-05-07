import os
import shutil


def move_images():
    # Get all file names in directory "images"
    files = os.listdir("images")
    # Create new directories "yellow" and "blue"
    os.mkdir("yellow")
    os.mkdir("blue")
    # Loop through all files
    for file in files:
        # Get the first 2 numbers of the file name
        first_number = int(file.split("_")[1])
        second_number = int(file.split("_")[2])
        # If the first number is bigger than the second, move the image to directory "yellow"
        if first_number > second_number:
            shutil.move("images/" + file, "yellow/" + file)
        # Otherwise, move the image to directory "blue"
        else:
            shutil.move("images/" + file, "blue/" + file)
