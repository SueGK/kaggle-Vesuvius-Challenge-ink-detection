import os
import cv2
import matplotlib.pyplot as plt
import tqdm

def read_image_mask(fragment_id):
    images = []
    idxs = range(from_slice,to_slice) # range(24, 40)
    for i in tqdm.tqdm(idxs):
        image = cv2.imread(base_path + f"/train/{fragment_id}/surface_volume/{i:02}.tif", 0)
        images.append(image)
    labels = cv2.imread(base_path + f"/train/{fragment_id}/inklabels.png", 0)
    mask = cv2.imread(base_path + f"/train/{fragment_id}/mask.png", 0)
    return images, labels, mask

def show_splits():
    for j in range(splits): 
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Create a figure with 1 row and 3 columns
        axes[0].imshow(images[(to_slice-from_slice)//2][j*splitter:(j+1)*splitter])
        axes[0].set_title('Image')
        axes[1].imshow(labels[j*splitter:(j+1)*splitter])
        axes[1].set_title('Labels')
        axes[2].imshow(mask[j*splitter:(j+1)*splitter])
        axes[2].set_title('Mask')
        plt.show()  # Show the figure

def save_split(dir_num):
    # Define the directory and create it if it does not exist
    dir_path = os.path.join(base_path, save_path, str(dir_num), "surface_volume")
    os.makedirs(dir_path, exist_ok=True)
    for i in range((to_slice-from_slice)):
        # Save the image
        if dir_num in (1,2,3):
            cv2.imwrite(os.path.join(dir_path, f"{i:01}.png"), images[i])
        else:
            cv2.imwrite(os.path.join(dir_path, f"{i:01}.png"), images[i][s*splitter:(s+1)*splitter])
    
    dir_path = os.path.join(base_path, save_path, str(dir_num))
    if dir_num in (1,2,3):
        cv2.imwrite(os.path.join(dir_path, "inklabels.png"), labels)
        cv2.imwrite(os.path.join(dir_path, "mask.png"), mask)
    else:
        cv2.imwrite(os.path.join(dir_path, "inklabels.png"), labels[s*splitter:(s+1)*splitter])
        cv2.imwrite(os.path.join(dir_path, "mask.png"), mask[s*splitter:(s+1)*splitter])


#==========> create splits <=============
from_slice = 24
to_slice = 40

base_path = '/root/autodl-tmp/VCInkDectection/input/vesuvius-challenge-ink-detection'
save_path = '/working/train_sub_png'
plot = True
save_full = True

# We split from 
for fragment_id in range(1,4):
    images, labels, mask = read_image_mask(fragment_id)
    if fragment_id == 1:
        fragment_start = 10
        splitter = 256 * 16
    if fragment_id == 2:
        fragment_start = 20
        splitter = 256 * 20
    if fragment_id == 3:
        fragment_start = 30
        splitter = 256 * 16
    splits = int(round(images[0].shape[0]/splitter))

    if plot:
        show_splits()
        
    for s in range(splits): 
        dir_num = fragment_start+s+1
        save_split(dir_num)
    if save_full:
        save_split(fragment_id)

