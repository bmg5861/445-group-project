import os
import json
import requests

def read_tracker_file():
    if os.path.exists('tracker.txt'):
        with open('tracker.txt', 'r') as f:
            last_image = f.read().strip()
        return last_image
    else:
        return None

def write_tracker_file(image_filename):
    with open('tracker.txt', 'w') as f:
        f.write(image_filename)

# Load image URLs and statuses from text file
with open('image_data.txt', 'r') as f:
    image_lines = f.readlines()

# Create folders if they don't exist
folders = ['withsignal', 'withoutsignal']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Read the last downloaded image from the tracker file
last_downloaded_image = read_tracker_file()

# Store downloaded images to check for duplicates
downloaded_images = set()

# Iterate through lines in the text file
for line in image_lines:
    data = json.loads(line.strip())
    url = data['waterfall']
    status = data['waterfall_status']
    folder_name = 'withsignal' if status == 'with-signal' else 'withoutsignal'
    filename = os.path.join(folder_name, url.split('/')[-1])  # Extract filename from URL
    
    # Check if the image was already downloaded or if it's a duplicate URL
    if last_downloaded_image and last_downloaded_image == filename:
        print(f"Skipping previously downloaded image: {filename}")
        continue
    
    if url in downloaded_images:
        print(f"Skipping duplicate image URL: {url}")
        continue
    
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Image downloaded: {filename}")
    
    # Update tracker file with the latest downloaded image
    write_tracker_file(filename)
    
    # Add the downloaded image URL to the set
    downloaded_images.add(url)