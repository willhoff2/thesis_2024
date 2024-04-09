###  Paleoclimate Preprocessing Script
### Last Update: 2-2
### Current Pipeline: 



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, ViTForImageClassification, ViTModel, ViTImageProcessor 

import torch
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.core.interactiveshell import InteractiveShell

import warnings


from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler

import os
import re
from sklearn.model_selection import GridSearchCV

import itertools
import torch.optim as optim

from datetime import date
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms.functional import to_tensor
from PIL import Image

from random import randint

InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', None)
warnings.filterwarnings('default')

#current_working_directory = os.getcwd()

# print output to the console
#print(current_working_directory)



# Function to clean column names
def clean_column_name(column_name):
    # Remove text within parentheses
    column_name = re.sub(r' \([^)]*\)', '', column_name)
    # Replace spaces with underscores
    column_name = column_name.replace(' ', '_')
    
    # Convert to lower case
    return column_name.lower().strip()

#### IMAGE PROCESSING START #####

class ImageData:
    def __init__(self, name, image):
        self.name = name
        self.image = image
        self.start = None
        self.end = None
        self.cm_to_p = None

class CustomDataset(Dataset):
    def __init__(self, indices, pixel_values, labels, depths, sources):
        self.indices = np.array(indices)
        # Convert each element in pixel_values to a tensor if it's not already
        self.pixel_values = torch.stack([pv.clone().detach() if isinstance(pv, torch.Tensor) else torch.tensor(pv, dtype=torch.float32) for pv in pixel_values])
        self.labels = labels
        self.depths = np.array(depths)
        self.sources = np.array(sources)

    def __len__(self):
        return len(self.pixel_values)

    def __getitem__(self, idx):
        index = self.indices[idx]
        image = self.pixel_values[idx]
        label = self.labels[idx]
        depth = self.depths[idx]
        source = self.sources[idx]
        return index, image, label, depth, source
    
    def concatenate(self, other_dataset, inplace = False):
        """ Concatenates another datase """
        if not isinstance(other_dataset, CustomDataset):
            raise ValueError('The other dataset must be an instance of CustomDataset')
        
        # Concatenate all elements
        new_indices = np.concatenate((self.indices, other_dataset.indices), 0)
        new_pixel_values =  torch.cat((self.pixel_values, other_dataset.pixel_values), 0)
        new_labels = torch.cat((self.labels, other_dataset.labels), 0)
        new_depths = np.concatenate((self.depths, other_dataset.depths), 0)
        new_sources = np.concatenate((self.sources, other_dataset.sources), 0)
        if inplace:
            self.indices = new_indices
            self.pixel_values = new_pixel_values
            self.labels = new_labels
            self.depths = new_depths
            self.sources = new_sources
        return CustomDataset(new_indices, new_pixel_values, new_labels,new_depths,new_sources)


def sortimg(img):
    return img.name

def load_and_display_images(folder_path, img_store, lake, display = False):
    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        # Construct full file path
        file_path = os.path.join(folder_path, file)
        if lake == "SVID":
            obj_name = file[14:23]
        elif lake == "LVID":
            obj_name = file[0]
        else:
            return "New Lake. Re-verify naming"
        #print(file, obj_name)

        # Check if the file is an image (you can add more extensions if needed)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            img = mpimg.imread(file_path)

            new_img = ImageData(name=obj_name, image=img)
            
            # Display the image
            if display:
                plt.imshow(img)
                plt.title(file)
                plt.show()
            
            img_store.append(new_img)


def crop_image(image_data, box, display = False):
    # Unpack the box coordinates
    left, lower, right, upper = box

    # Crop the image using array slicing
    cropped_image = image_data.image[lower:upper, left:right]

    # Update image data
    cropped = cropped_image

    # Optionally, display the cropped image
    if display:
        plt.imshow(cropped)
        plt.axis('off')  # This hides the axis
        plt.show()

    return cropped

def add_black_chunk(image, start_cm, end_cm):
    """
    Takes an image and blacks out the pixels from start_cm to end_cm.
    Each cm is 200 pixels wide, image is assumed to be 1300 pixels tall.
    """
    # Validate inputs
    if start_cm < 0 or end_cm > 5 or start_cm > end_cm:
        raise ValueError("Start and end cm must be within 0-5 and start_cm <= end_cm")

    # Make a writable copy of the image
    image = np.array(image, copy=True)

    # Calculate start and end pixels based on the cm inputs
    start_px = start_cm * 200  # Convert cm to pixels
    end_px = end_cm * 200      # Convert cm to pixels

    # Adjust end_px if it exceeds the image's width
    if end_cm == 5:  # Assuming 4 means the end of the image
        _, width, _ = image.shape  # Get the width of the image
        end_px = width  # Use the image's width as the end pixel

    # Black out the specified section
    image[:, start_px:end_px] = 0  # Set the pixels to black

    return image
 


def maybe_add_black_chunk(image):
    # Make a writable copy of the image array
    image = np.array(image, copy=True)  # This ensures the array is writable
    
    # Your existing logic for maybe adding a black chunk
    # For example, if the logic decides to modify the image:
    # Determine the width for random start of black chunk
    width = image.shape[1]
    
    # Randomly decide if adding a black chunk
    if np.random.rand() <= 0.05:  # 5% chance
        start_cm = np.random.randint(0, 4)  # Random start position 0-3
        end_cm_decision = np.random.randint(1, 101)  # Random end width decision
        
        # Calculate start and end pixel positions based on start_cm
        start_px = start_cm * 200  # Each cm is 200px wide
        
        # Determine the end pixel based on the random decision
        if end_cm_decision <= 40:
            end_px = start_px + 200  # 1cm
        elif end_cm_decision <= 80:
            end_px = start_px + 400  # 2cm
        else:
            end_px = start_px + 600  # 3cm
        
        # Ensure the end pixel does not exceed image width
        end_px = min(end_px, width)
        
        # Modify the image
        image[:, start_px:end_px] = 0  # Set the selected chunk to black

    return image


def transform_images(images, feature_extractor):
    inputs = feature_extractor(images=images, return_tensors="pt")
    return inputs['pixel_values']

def transform_image(filepath, feature_extractor = None, read = True, transform = True):
    # Read the image
    if read:
        image=filepath
    else:
        image = mpimg.imread(filepath)

    

    
    if transform:
        # Convert the image to a writable NumPy array if necessary
        if not image.flags.writeable:
            image = np.array(image).copy()
        # Convert the image to a PyTorch tensor
        image_tensor = to_tensor(image)  # Converts to tensor and scales to [0, 1]

        # The feature extractor expects a batch of images, so add an extra batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        inputs = feature_extractor(images=image_tensor, return_tensors="pt", do_rescale=False)  # do_rescale set to False
        return inputs['pixel_values'].squeeze(0)

    else:
        if image.dtype == np.float32:  # Assuming the image is in [0, 1] for float types
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        
        # Step 3: Define the transformation pipeline
        transform_pipeline = Compose([
            Resize((224, 224)),  # Resize the image to 224x224 pixels
            ToTensor(),  # Converts to tensor and scales to [0, 1]
        ])
        
        # Step 4: Apply the transformations
        image_tensor = transform_pipeline(image_pil)
        
        # Return the transformed image tensor
        return image_tensor
    
    
    
                
        

## Cropping Unnecesary parts
# Left, top, right, bottom

## Cropping Unnecesary parts
# Left, top, right, bottom
def crop_svid(display = False, do_print = False, return_full = False):
    full_SVID = []

    #### Update here with local path for image directory #####
    image_dir = "/Users/willhoff/Desktop/thesis_2024/data/img_data/SVID"
    load_and_display_images(image_dir ,full_SVID, "SVID")
    full_SVID.sort(key=sortimg,reverse=False)
 
    cropped_SVID = []

    whole = 0

    ## This is only guessing, do something better potentially. More fine grain
    ## 1. 1000 0 31275 1300
        ## 6-157 cm
    start_cm = 6
    end_cm = 157
    start = 1000
    end = 31275

    box = (start, 0, end, 1300)

    full_SVID[0].start = 1000
    full_SVID[0].end = 31275
    full_SVID[0].cm_to_p = (end - start)/(end_cm-start_cm)

    cropped_image = crop_image(full_SVID[0], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[0].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 1 ratio {(end - start)/(end_cm-start_cm)}")




    ## This is only guessing, do something better potentially. More fine grain
    ## 2. 800 0 31290 1300
        ## 5-157 cm
    start_cm = 5
    end_cm = 157
    start = 800
    end = 31290

    box = (start, 0, end, 1300)

    full_SVID[1].start = start
    full_SVID[1].end = end
    full_SVID[1].cm_to_p = (end - start)/(end_cm-start_cm)

    cropped_image = crop_image(full_SVID[1], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[1].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 2 ratio {(end - start)/(end_cm-start_cm)}")


    ## This is only guessing, do something better potentially. More fine grain
    ## 3. 800 0 31100 1300
        ## 5-156 cm
    start_cm = 5
    end_cm = 156
    start = 800
    end = 31100

    box = (start, 0, end, 1300)

    full_SVID[2].start = start
    full_SVID[2].end = end
    full_SVID[2].cm_to_p = (end - start)/(end_cm-start_cm)

    cropped_image = crop_image(full_SVID[2], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[2].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 3 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 4. 1200 0 28000 1300
        ## 7-140.5 cm
    start_cm = 7
    end_cm = 140.5
    start = 1200
    end = 28000

    box = (start, 0, end, 1300)

    full_SVID[3].start = start
    full_SVID[3].end = end
    full_SVID[3].cm_to_p = (end - start)/(end_cm-start_cm)

    cropped_image = crop_image(full_SVID[3], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[3].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 4 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 5. 1400 0 31700 1300
        ## 7-159 cm (8-159)
    

    start_cm = 8
    end_cm = 159
    start = 1400
    end = 31700

    box = (start, 0, end, 1300)

    full_SVID[4].start = 1200
    full_SVID[4].end = 31700
    full_SVID[4].cm_to_p = (end - start)/(end_cm-start_cm)
    

    cropped_image = crop_image(full_SVID[4], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[4].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 5 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 6. 875 0 22325 1300
        ## 5.5-112 cm 
    

    start_cm = 5.5
    end_cm = 112
    start = 875
    end = 22325

    box = (start, 0, end, 1300)

    full_SVID[5].start = start
    full_SVID[5].end = end
    full_SVID[5].cm_to_p = (end - start)/(end_cm-start_cm)
    

    cropped_image = crop_image(full_SVID[5], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[5].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 6 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 7. 1600 0 16750 1300
        ## 9-83.5 cm
    

    start_cm = 9
    end_cm = 83.5
    start = 1600
    end = 16625

    box = (start, 0, end, 1300)

    full_SVID[6].start = start
    full_SVID[6].end = end
    full_SVID[6].cm_to_p = (end - start)/(end_cm-start_cm)

    cropped_image = crop_image(full_SVID[6], box, display=display)
    cropped_SVID.append(ImageData(full_SVID[6].name,cropped_image))
    whole += (end_cm-start_cm)

    if do_print: 
        print(f"img 7 ratio {(end - start)/(end_cm-start_cm)}")
        print(f"whole measure {whole}")

    if return_full:
        return cropped_SVID, full_SVID
    else:
        return cropped_SVID


## Cropping Unnecesary parts
# Left, top, right, bottom

def crop_lvid(display = False, do_print = False, return_full = False):
    full_LVID = []
    cropped_LVID = []
    load_and_display_images('img_data/LVID',full_LVID, "LVID")
    ## reverse = False because 1B-7B seems darkest so would make sense for it to be on the bottom
    full_LVID.sort(key=sortimg,reverse=False)   

    whole = 0
## This is only guessing, do something better potentially. More fine grain
## 1. 3200 0 31075 1300
    ## 17-156 cm  

    start_cm = 17
    end_cm = 156
    start = 3200
    end = 31075
    box = (start, 0, end, 1500)

    full_LVID[0].start = start
    full_LVID[0].end = end
    full_LVID[0].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[0], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[0].name,cropped_image))
    if do_print:
        print(f"img 1 ratio {(end - start)/(end_cm-start_cm)}")


    ## This is only guessing, do something better potentially. More fine grain
    ## 2. 1400 0 15525 1300
        ## 8-78 cm 
    start_cm = 8
    end_cm = 78
    start = 1400
    end = 15525
    box = (start, 0, end, 1300)
    full_LVID[1].start = start
    full_LVID[1].end = end
    full_LVID[1].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[1], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[1].name,cropped_image))
    if do_print:
        print(f"img 2 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 3. 1200 0 21850 1300
        ## 7-109.5 cm
    start_cm = 7
    end_cm = 109.5
    start = 1200
    end = 21825
    box = (start, 0, end, 1300)

    full_LVID[2].start = start
    full_LVID[2].end = end
    full_LVID[2].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[2], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[2].name,cropped_image))
    if do_print:
        print(f"img 3 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 4. 1000 0 21650 1300 --> 1000 0 14550 1300 (15750 to 79, 14550 to 73)
        ## 6-108.5 cm (6-108.5) --> NOW DOWN TO 73 W OVERLAP 6-73
    start_cm = 6
    end_cm = 73
    start = 1000
    end = 14550
    box = (start, 0, end, 1300)

    full_LVID[3].start = start
    full_LVID[3].end = end
    full_LVID[3].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[3], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[3].name,cropped_image))
    if do_print:
        print(f"img 4 ratio {(end - start)/(end_cm-start_cm)}")


    ## This is only guessing, do something better potentially. More fine grain
    ## 5. 1000 0 21125 1300 --> 2490 0 21125 1300
        ## 6-106 cm (matches) --> 12.5 (12.5 cm deep) to 106
    start_cm = 12.5
    end_cm = 106
    start = 2490
    end = 21125
    box = (start, 0, end, 1300)

    full_LVID[4].start = start
    full_LVID[4].end = end
    full_LVID[4].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[4], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[4].name,cropped_image))
    if do_print:
        print(f"img 5 ratio {(end - start)/(end_cm-start_cm)}")

    ## This is only guessing, do something better potentially. More fine grain
    ## 6. 900 0 21900 1300
        ## 5.5-109.5 cm (matches)
    start_cm = 5.5
    end_cm = 109.5
    start = 900
    end = 21825
    box = (start, 0, end, 1300)

    full_LVID[5].start = start
    full_LVID[5].end = end
    full_LVID[5].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[5], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[5].name,cropped_image))
    if do_print:
        print(f"img 6 ratio {(end - start)/(end_cm-start_cm)}")

    ## 7. 900 0 23400 1300
        ## 5.5-117.5 cm (5.5-117.5)
    start_cm = 5.5
    end_cm = 117.5
    start = 900
    end = 23400
    box = (start, 0, end, 1300)

    full_LVID[6].start = start
    full_LVID[6].end = end
    full_LVID[6].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[6], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[6].name,cropped_image))
    if do_print:
        print(f"img 7 ratio {(end - start)/(end_cm-start_cm)}")

    ## 8. 1200 0 10850 1300
        ## 7-54.5 cm

    start_cm = 7
    end_cm = 54.5
    start = 1200
    end = 10850
    box = (start, 0, end, 1300)

    full_LVID[7].start = start
    full_LVID[7].end = end
    full_LVID[7].cm_to_p = (end - start)/(end_cm-start_cm)
    whole += (end_cm-start_cm)

    cropped_image = crop_image(full_LVID[7], box, display= display)
    cropped_LVID.append(ImageData(full_LVID[7].name,cropped_image))
    if do_print: 
        print(f"img 8 ratio {(end - start)/(end_cm-start_cm)}")
        print(f"whole measure {whole}")
    if return_full:
        return cropped_LVID, full_LVID
    else: 
        return cropped_LVID

############# IMAGE PROCESSING DONE #################







    
############## DATASET CONSTRUCTION ##################

def build_and_save_MA2(lake_name, sediment_width=5, prediction_size = 1, debug = False):
    '''
    Builds and saves a dataframe to memory. Returns dataframe
    Params:
        lake_name: Lake name to be used as source value (str)
        sediment_width: Width for image data (float)
        prediction_size: Size of prediction value (int)
        debug: helpful prints (bool)
    '''
    
    ## Still need:
        ## wraparound
        ## Standardized test areas
    if lake_name == "LVID":
        _, images = crop_lvid(return_full=True)
    elif lake_name == "SVID":
        _, images = crop_svid(return_full=True)
    else:
        print(f"lake {lake_name} crop function not implemented")
        return
    
    br_df = pd.read_csv(f"../data/{lake_name}_brGDGTs.csv")
    geochem_df = pd.read_csv(f"../data/{lake_name}_other.csv")
    depth_col = "sediment_depth"

    ## Find a way to make this variable based on what dataset were creating
        ## Could load both in and make it one dataset where we have all 4 labels: mbt, bsi, toc, c-n
        ## We will only adjust to extract certain column
        ## To find known v unknown, we will just look at column values
    data = {
        'index': [],
        'depth': [],
        '%toc': [],
        'mbt': [],
        'bsi': [],
        'c-n': [],
        'chunk': [],
        'source': []
    }

    chunk_save_dir = f'MA_data/{lake_name}/chunks'
    os.makedirs(chunk_save_dir, exist_ok=True)
    data_save_dir = f'MA_data/{lake_name}/data'
    os.makedirs(data_save_dir, exist_ok=True)
    data_filename = f"{lake_name}_dataset.csv"
    save_path = os.path.join(data_save_dir, data_filename)

    j = 0  # Chunk index
    ## Start at 5 since our first edge is at cm 5 (Measurements are taken at the end not beginning, first chunk will be [0,5) )
    full_depth = 5

    ## Target values: Make this param and check for brgdgt
    vals = ['mbt', 'bsi','c-n', '%toc']


    for i, img_obj in enumerate(images):

        ## Extracting Image specific data
        chunk_size = int(img_obj.cm_to_p * sediment_width)
        start_pixel = img_obj.start
        if i != 0:
            end_pixel = start_pixel + int(img_obj.cm_to_p * prediction_size)
        else:
            end_pixel = start_pixel + chunk_size
        p_to_cm = 1/img_obj.cm_to_p

        ## Add on addition to front of img_obj
        if i!= 0:
           
            ## Be careful about Incrementing by correct for full and end (use prev_cm_to_p)
            wrap_count = 0
            wrap_start = 0

            ## Append 
            #img_obj.image = np.concatenate((wrap_around, img_obj.image), axis=1)

            

        while end_pixel <= img_obj.end:

            ## We have to switch our condition to be opposite now that full_depth measures the edge of each chunk

            ## Adding in all label values
            for target_col in vals:
                if target_col == "mbt":
                    section_data =  br_df[( br_df[depth_col] <= full_depth) & 
                            ( br_df[depth_col] > (full_depth-sediment_width))]
                    average_target = section_data[target_col].mean() if not section_data.empty else np.nan
                else:
                    section_data =  geochem_df[( geochem_df[depth_col] <= full_depth) & 
                            ( geochem_df[depth_col] > (full_depth-sediment_width))]
                    average_target = section_data[target_col].mean() if not section_data.empty else np.nan

                data[target_col].append(average_target)

                ## Print here to see
                if debug:
                    print(f"Depth {full_depth-5} - {full_depth} \n")
                    print(section_data[[depth_col, target_col]].head())
                    print(f"Target {target_col} {average_target}\n")
            
            if i != 0 and wrap_count < 4:
                wrap_chunk = wrap_around[0:1300, wrap_start:wrap_end]
                chunk = img_obj.image[0:1300, start_pixel:end_pixel]
                chunk = np.concatenate((wrap_chunk, chunk), axis=1)
                wrap_start += int(prev_cm_to_p * prediction_size)
                wrap_count += 1
            else:
                chunk = img_obj.image[0:1300, start_pixel:end_pixel] 

                ## Incrementing start here  
                start_pixel += int(img_obj.cm_to_p * prediction_size) 
             
            filename = f'chunk_{j}.jpg'
            image_save_path = os.path.join(chunk_save_dir, filename)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, chunk)
            else:
                print(f"Image '{filename}' already exists. Skipping overwrite.")
            
            data['index'].append(j)
            ## Shoudld 0-5 have average of 2.5
            data['depth'].append(full_depth - ((sediment_width)/ 2))  # Center depth of the chunk
            data['chunk'].append(image_save_path)
            data['source'].append(lake_name)

            # This is to make sure were iterating through 1 cm of mud no matter what, even if from last image
            # if i != 0 and wrap_count < 4:
            #     print("cm check", prev_cm_to_p, img_obj.cm_to_p)
            #     start_pixel += int(prev_cm_to_p * prediction_size)
                
            # else:
                
            
            end_pixel += int(img_obj.cm_to_p * prediction_size) ## moving along 1 cm at a time

            j += 1

            ## Will never be in an in between space, because if its too large while loop won't run
            full_depth += prediction_size

        
        ## We need this check to ensure depths match up from potential smaller than sediment_width size chunk not being added on
        ## end_pixel will always be farther! That's the end of the loop
        #full_depth += int((img_obj.end - (end_pixel - chunk_size)) * p_to_cm)

        ## Wrapping around last 4 cm
        wrap_cm_width = 4 * img_obj.cm_to_p
        prev_cm_to_p = img_obj.cm_to_p
        last_4 = int(img_obj.end - wrap_cm_width)
        if debug:
            print("bounds", last_4, img_obj.end)
        wrap_around = img_obj.image[0:1300, last_4:img_obj.end]
        wrap_end = img_obj.end


    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

    return df             

def attach_labels(target,sample_df, images, depth_col, lake_name, scaled=False, sediment_width = 5):
    
    scaler = None
    if scaled:
        scaler = StandardScaler()
        target_col = target + '_scaled'
        sample_df[target_col] = scaler.fit_transform(sample_df[[target]])
    else:
        target_col = target
        
        
    
    ## Pixels to cm
        ## To make sure our depths are accurate
    p_to_cm = 5/997
    cm_to_p = 997/5 

    full_depth = 0
    chunk_size = int(cm_to_p * sediment_width)
    chunks = []
    chunk_labels = []

    j = 0
    indices = []

    depths = []
    sources = []

    for i,object in enumerate(images):

        full_end = object.end
        start_pixel = object.start
        end_pixel = object.start + chunk_size
        while end_pixel <= full_end:

            ## Adding on labels
            section_data =  sample_df[( sample_df[depth_col] >= full_depth) & 
                            ( sample_df[depth_col] < (full_depth+sediment_width))]

            # Calculate the average target value for the segment
            average_target = section_data[target_col].mean()

            if len(section_data) > 0:
                chunk_labels.append(average_target)
                chunk = object.image[0:1300, start_pixel:end_pixel]

                ## Depth of image is from range full_depth to full_depth + sediment width
                ## Although it may be more correct to average the target depth values, for ease of reproduction,
                ## changing from section_data[depth_col].mean() to midpoint of image
                chunks.append(chunk) 
                if (end_pixel + chunk_size) > full_end:
                    depth_value = (full_end - ((end_pixel - chunk_size) * p_to_cm / 2))
                    depths.append(depth_value)
                else: 
                    depth_value = full_depth + (sediment_width/2)
                    depths.append(depth_value)
                
                sources.append(lake_name)
                indices.append(j)

            ## While loop Increments
            start_pixel += chunk_size
            end_pixel += chunk_size
            j += 1
            indices.append(j)
            ## Will never be in an in between space, because if its too large while loop won't run
            full_depth += sediment_width

        ## We need this check to ensure depths match up from potential smaller than sediment_width size chunk not being added on
        if end_pixel > full_end:
            full_depth += (full_end - (end_pixel - chunk_size)) * p_to_cm
                
                
     # Convert lists to numpy arrays
    images_array = np.array(chunks)
    labels_array = np.array(chunk_labels)

    # Create boolean masks
    nan_mask = np.isnan(labels_array)
    non_nan_mask = ~nan_mask

    # Filter the arrays
    images_test = images_array[nan_mask]
    images_known = images_array[non_nan_mask]
    labels_test = labels_array[nan_mask]
    labels_known = labels_array[non_nan_mask]
    
    # Assuming pixel_values already a tensor in the shape [num_images, channels, height, width]
    pixel_values_tensor = transform_images(images_known)
    labels_tensor = torch.tensor(labels_known, dtype=torch.float32)
    
    ## Formerly returned inputs: return pixel_values_tensor, labels_tensor, inputs, scaler, np.array(depths), np.array(sources)
    return images_known, labels_tensor, scaler, np.array(depths), np.array(sources), np.array(indices)

def build_and_save_MA(target, sample_df, images, depth_col, lake_name, scaled=False, sediment_width=5, test_size=0.2, random=True, random_state=2, prediction_size = 1):
    '''
    Builds and saves a dataframe to memory. Returns dataframe
    Params:
        dataset: Specifies whether you want geochem or brGDGT (str)
        images: Image set for chunks (list)
        lake_name: Lake name to be used as source value (str)
        scaled: boolean value specifying whether labels should be scaled (bool)
        sediment_width: Width for target and image data (float)
        test_size: Fraction value between 0-1 for test size (float)
        random: boolean value whether test set should be sequential or random (bool)
        random_state: random state to be used if test set is random (int)
    '''
    
    ## Still need:
        ## wraparound
        ## Standardized test areas
    
    if target == "mbt":
        sample_df = pd.load_csv(f".../data/{lake_name}_brGDGTs.csv")
    else:
        sample_df = pd.load_csv(f".../data/{lake_name}_other.csv")
    depth_col = "sediment_depth"

    scaler = None

    if scaled:
        scaler = StandardScaler()
        target_col = f'{target}_scaled'
        sample_df[target_col] = scaler.fit_transform(sample_df[[target]])
    else:
        target_col = target

    data = {
        'index': [],
        'depth': [],
        'label': [],
        'chunk': [],
        'source': [],
        'set': [],  # This will initially mark all data as "N/A" to be updated later <-- dont leave any NA
        'known' : [] # Fill in w true or false if we know v don't know
    }

    test_data = {
        'depth': [],
        'label': [],
        'source': []
    }



    chunk_save_dir = f'MA_chunk_data/{lake_name}/{sediment_width}cm/chunks/{target}'
    os.makedirs(chunk_save_dir, exist_ok=True)
    data_save_dir = f'MA_chunk_data/{lake_name}/{sediment_width}cm/data'
    os.makedirs(data_save_dir, exist_ok=True)
    data_filename = f"{target}_dataset.csv"
    save_path = os.path.join(data_save_dir, data_filename)
    test_depth_filename = f"{target}_test_depth_dataset.csv"
    test_depth_save_path = os.path.join(data_save_dir, test_depth_filename)

    j = 0  # Chunk index
    ## Start at 5 since our first edge is at cm 5
    full_depth = 5

    ## Moving out of for loop
    base_start = 0
    test_1_start, test_1_end, test_2_start, test_2_end = generate_random_test_intervals(base_start)

    ## Save depth values here
    section_data =  sample_df[((sample_df[depth_col] >= test_1_start) & (sample_df[depth_col] <= test_1_end)) |
                    ((sample_df[depth_col] >= test_2_start) & (sample_df[depth_col] <= test_2_end))]
    # Convert the relevant columns to lists and append them to the test_data dictionary
    test_depths = section_data[depth_col].tolist()
    test_labels = section_data[target_col].tolist()

    # Append each item in the lists to the corresponding keys in the test_data dictionary
    for depth, label in zip(test_depths, test_labels):
        test_data['depth'].append(depth)
        test_data['label'].append(label)
        test_data['source'].append(lake_name)  # Appending lake_name directly since we're iterating

    
    ## Not just state = "Train"
    in_test = (full_depth >= test_1_start and full_depth <= test_1_end) or (full_depth >= test_2_start and full_depth <= test_2_end)
    ## Initial state
    if in_test:
        state = "Test"
    else:
        state = "Train"

    for i, img_obj in enumerate(images):

        ## Extracting Image specific data
        chunk_size = int(img_obj.cm_to_p * sediment_width)
        start_pixel = img_obj.start
        end_pixel = start_pixel + chunk_size
        p_to_cm = 1/img_obj.cm_to_p
        
        ## We need to check state, not just state = "Train
        #state = "Train" --> nO point to reset at end of image, if its still in test, continue as before


        while end_pixel <= img_obj.end:

            ## check if full_depth within random range:
            ## If its test: we start anew w the newest added cm. end_pixel now equals start pixel, end_pixel now equals start pixel + 5 cm
                    ## We continue to shuffle until end_pixel gets outside of range
                        ## Once outside test range, we continue with end_pixel as start pixel and end_pixel as start_pixel + 5 cm
                            ##  (make sure there isnt back to back test chunks if there is, maybe change!)
                ## Within these, if we have unknown chunks we just save as unknown, maybe new column with first train v test then unknown v known
            
            if full_depth >= base_start + 100:
                base_start += 100
                test_1_start, test_1_end, test_2_start, test_2_end = generate_random_test_intervals(base_start)


                ## Save depth values here
                section_data =  sample_df[((sample_df[depth_col] >= test_1_start) & (sample_df[depth_col] <= test_1_end)) |
                         ((sample_df[depth_col] >= test_2_start) & (sample_df[depth_col] <= test_2_end))]
                # Convert the relevant columns to lists and append them to the test_data dictionary
                test_depths = section_data[depth_col].tolist()
                test_labels = section_data[target_col].tolist()

                # Append each item in the lists to the corresponding keys in the test_data dictionary
                for depth, label in zip(test_depths, test_labels):
                    test_data['depth'].append(depth)
                    test_data['label'].append(label)
                    test_data['source'].append(lake_name)  # Appending lake_name directly since we're iterating

                

                


            in_test = (full_depth >= test_1_start and full_depth <= test_1_end) or (full_depth >= test_2_start and full_depth <= test_2_end)
            if in_test:
                if state == "Train":
                    state = "Test"
                    ## This doesnt seem correct because full depth is an average measure, not a point measure. We dont want the middle, we want the edge
                        ## Full depth is fine because we are incrementing every time by 1 cm in order to be on the edge
                    ## We jumped now from a certain point
                    ## Make sure to get the whole new pixel
                    start_pixel = (end_pixel - int(img_obj.cm_to_p * prediction_size))
                    end_pixel = start_pixel + chunk_size
                    if end_pixel > img_obj.end:
                        break
                    ## We just skipped 4 ahead, now the end of our core, is at the end of our chunk
                    full_depth += 4

                    ## No elif as if its test, we dont need to reset and we can continue as before
                
                
            ## Since its elif, we know were not in test set anymore
            elif state == "Test":
                state = "Train"
                start_pixel = (end_pixel - int(img_obj.cm_to_p * prediction_size))
                end_pixel = start_pixel + chunk_size
                if end_pixel > img_obj.end:
                    break
                ## We just skipped 4 ahead, now the end of our core, is at the end of our chunk
                full_depth += 4

            
            ## If its train/test: we continue as before  

            ## We have to switch our condition to be opposite now that full_depth measures the edge of each chunk
            section_data =  sample_df[( sample_df[depth_col] <= full_depth) & 
                            ( sample_df[depth_col] > (full_depth-sediment_width))]
            
            ## Print here to see
            print(f"Depth {full_depth-5} - {full_depth} \n")
            print(section_data[[depth_col, target_col]].head())

            average_target = section_data[target_col].mean() if not section_data.empty else np.nan
            chunk = img_obj.image[0:1300, start_pixel:end_pixel]

            print(f"Target {average_target}\n")

            filename = f'chunk_{j}.jpg'
            image_save_path = os.path.join(chunk_save_dir, filename)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, chunk)
            else:
                print(f"Image '{filename}' already exists. Skipping overwrite.")
            
            data['index'].append(j)
            data['depth'].append(full_depth - ((sediment_width - 1)/ 2))  # Center depth of the chunk
            data['label'].append(average_target)
            data['chunk'].append(image_save_path)
            data['source'].append(lake_name)
            data['set'].append(state)  
            if np.isnan(average_target):
                data['known'].append(False) 
            else:
                data['known'].append(True) 

            start_pixel += int(img_obj.cm_to_p * prediction_size)
            end_pixel += int(img_obj.cm_to_p * prediction_size) ## moving along 1 cm at a time
            j += 1
            ## Will never be in an in between space, because if its too large while loop won't run
            full_depth += prediction_size

        
        ## We need this check to ensure depths match up from potential smaller than sediment_width size chunk not being added on
        ## end_pixel will always be farther! That's the end of the loop
        full_depth += int((img_obj.end - (end_pixel - chunk_size)) * p_to_cm)

    df = pd.DataFrame(data)
    df_test_depth = pd.DataFrame(test_data)

    # Save the test depth dataset
    df_test_depth.to_csv(test_depth_save_path, index=False)

    df.to_csv(save_path, index=False)

    return df

def generate_random_test_intervals(base_start):
    """
    Generate two distinct 10 cm intervals as test set from a given 100 cm range starting from base_start.

    Parameters:
    - base_start: The starting cm of the 100 cm range to consider for generating test intervals.

    Returns:
    - A tuple of two tuples, each representing the start and end cm of the test intervals.
    """
    # Generate two distinct random indices between 0 and 9
    test_1 = randint(0, 9)
    test_2 = randint(0, 9)
    while test_2 == test_1:
        test_2 = randint(0, 9)

    # Calculate the actual start and end cm for each test interval based on the base_start
    test_1_start = base_start + test_1 * 10  # Convert index to cm by multiplying by 10 and adding to base_start
    test_1_end = test_1_start + 9  # 10 cm interval, so end is start + 9

    test_2_start = base_start + test_2 * 10
    test_2_end = test_2_start + 9

    return test_1_start, test_1_end, test_2_start, test_2_end

def build_and_save_dataset(target,sample_df, images, depth_col, lake_name, scaled=False, sediment_width = 5, test_size = 0.2, random = True, random_state = 2):
    '''
    Builds and saves a dataframe to memory. Returns dataframe
    Params:
        target: target value [] (str)
        sample_df: DataFrame to pull label data from (DataFrame)
        images: Image set for chunks (list)
        depth_col: Title of depth Column (str)
        lake_name: Lake name to be used as source value (str)
        scaled: boolean value specifying whether labels should be scaled (bool)
        sediment_width: Width for target and image data (float)
        test_size: Fraction value between 0-1 for test size (float)
        random: boolean value whether test set should be sequential or random (bool)
        random_state: random state to be used if test set is random (int)
    '''
    scaler = None
    if scaled:
        scaler = StandardScaler()
        target_col = target + '_scaled'
        sample_df[target_col] = scaler.fit_transform(sample_df[[target]])
    else:
        target_col = target
        
        
    

    full_depth = 0
    chunks = []
    chunk_labels = []

    j = 0
    indices = []

    depths = []
    sources = []

    data = {
        'index': [],
        'depth': [],
        'label': [],
        'chunk': [],
        'source': [],
        'set': []
    }

    ## Save Location
    chunk_save_dir = f'chunk_data/{lake_name}/{sediment_width}cm/chunks'
    os.makedirs(chunk_save_dir, exist_ok=True)
    data_save_dir = f'chunk_data/{lake_name}/{sediment_width}cm/data'
    os.makedirs(data_save_dir, exist_ok=True)

    # Check if CSV already exists
    data_filename = f"{target}_dataset.csv"
    save_path = os.path.join(data_save_dir, data_filename)

    
    for i,object in enumerate(images):
        ## Each Object has: 
            # self.name: name for the image
            # self.image: image data
            # self.start: Start pixel
            # self.end: end pixel
            # self.cm_to_p: Conversion factor to go from cm to pixels for a given image
        
        ## Defining image specific data
        p_to_cm = 1/object.cm_to_p
        cm_to_p = object.cm_to_p
        full_end = object.end
        start_pixel = object.start
        chunk_size = int(cm_to_p * sediment_width)
        end_pixel = object.start + chunk_size
        
        while end_pixel <= full_end:

            ## Adding on labels
            section_data =  sample_df[( sample_df[depth_col] >= full_depth) & 
                            ( sample_df[depth_col] < (full_depth+sediment_width))]

            # Calculate the average target value for the segment
            average_target = section_data[target_col].mean()
            chunk = object.image[0:1300, start_pixel:end_pixel]

            ## Only append if actual data??
            data['label'].append(average_target)

            ## Calculating chunk depth value
            depth_value = full_depth + (sediment_width/2)

            filename = f'chunk_{j}.jpg'
            image_save_path = os.path.join(chunk_save_dir, filename)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, chunk)
            else:
                print(f"Image '{filename}' already exists. Skipping overwrite.")

            data['chunk'].append(image_save_path) 
            data['depth'].append(depth_value)  
            data['source'].append(lake_name)
            data['index'].append(j)
            data['set'].append("N/A")

            ## While loop Increments
            start_pixel += chunk_size
            end_pixel += chunk_size
            j += 1
            ## Will never be in an in between space, because if its too large while loop won't run
            full_depth += sediment_width

        ## We need this check to ensure depths match up from potential smaller than sediment_width size chunk not being added on
        ## end_pixel will always be farther! That's the end of the loop
        full_depth += int((full_end - (end_pixel - chunk_size)) * p_to_cm)

    
    # Convert lists to numpy arrays (if necessary)
    labels_array = np.array(data['label'])

    # Split indices into train and test, only where labels are not NaN
    non_nan_indices = [i for i, label in enumerate(labels_array) if not np.isnan(label)]
    if random:
        train_indices, test_indices = train_test_split(non_nan_indices, test_size=test_size, random_state=random_state)
    else:
        split_idx = int((1 - test_size) * len(non_nan_indices))
        train_indices = non_nan_indices[:split_idx]
        test_indices = non_nan_indices[split_idx:]

    # Assign "Train" or "Test" to the Set column based on the indices
    for idx in train_indices:
        data['set'][idx] = "Train"
    for idx in test_indices:
        data['set'][idx] = "Test"

    ## Made as dataframe
    df = pd.DataFrame(data) 

    df.to_csv(os.path.join(data_save_dir, data_filename), index = False)

    return df


                    
#################### DATALOADER/TEST/TRAIN CREATION #######################    


def get_fold_test_data(lake_name, target, fold):
    '''
    Creates fold data with fold specified as test set. Returns test lists and test dataframe.
    lake_name: lake that image and label data should come from [SVID, LVID, both] (str)
    target: target value for label data (str)
    fold: Specifies which fold as test set. [0,1,2,3,4] (int)
    '''
    depth_col = "sediment_depth"
    
    lake_name = lake_name.upper()
    data_path = f'MA_data/{lake_name}/data/{lake_name}_dataset.csv'
    df = pd.read_csv(data_path)

    if target == 'mbt':
        test_1cm_df = pd.read_csv(f"../data/{lake_name}_brGDGTs.csv")
    else:
        test_1cm_df = pd.read_csv(f"../data/{lake_name}_other.csv")

    fold_length = int(df.shape[0]/5)

    next_start = fold * fold_length
    next_end = next_start + fold_length

    test_data_folder = f"/Users/willhoff/Desktop/research_23_24/paleoclimate/will_sandbox/MA_data/{lake_name}/fold_{fold}/test_data/"
    test_img_save_dir = f"/Users/willhoff/Desktop/research_23_24/paleoclimate/will_sandbox/MA_data/{lake_name}/fold_{fold}/test_chunks/"
    os.makedirs(test_data_folder, exist_ok=True)
    os.makedirs(test_img_save_dir, exist_ok = True)

    state = "test" if next_start == 0 else "train"
    cm = [0,1,2,3,4,5]

    test_indices, test_pixels, test_labels, test_depths, test_sources = [],[],[],[],[]
    ## Initial test_data
    filtered_test_data =  test_1cm_df[((test_1cm_df[depth_col] >= next_start) & 
                                  (test_1cm_df[depth_col] < next_end) & 
                                  pd.notna(test_1cm_df[target]))].copy()
                
    filtered_test_data['source'] = lake_name


    for i, row in df.iterrows(): ## for data:
        ## If its in position 5, the opposite data will have a black chunk for the whole thing
        if next_start not in cm[1:5] and next_end not in cm[1:5] and state == "test":
                
            test_indices.append(i)
            test_labels.append(row[target])
            test_depths.append(row['depth'])
            test_sources.append(row['source'])

            ## Use maybe_add_black_box function that adds a random 1 cm black box into the image with probability 5%
            image = mpimg.imread(row['chunk'])
            image = maybe_add_black_chunk(image)
            
            ## Transform later so I can save as CSV and not worry about pickle
            #pixel_img = transform_image(image,feature_extractor)

            filename = f'chunk_{i}.jpg'
            image_save_path = os.path.join(test_img_save_dir, filename)
            test_pixels.append(image_save_path)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, image)
            else:
                #print(f"Image '{filename}' already exists. Skipping overwrite.")
                pass

        ## Break in cm --> We also need to increment state somewhere
        elif next_start in cm[1:5]:
            #print("Check", cm[1:5], i)
            #print("Check", cm[1:6], i)
            #print(f"break in cm {next_start} of {cm}, state {state}, image {i} \n")

            test_indices.append(i)
            test_labels.append(row[target])
            test_depths.append(row['depth'])
            test_sources.append(row['source'])

            ## Take cm from image split to end and keep it. concatenate black square of size start cm[0] - break to front
            image = mpimg.imread(row['chunk'])
            cm_index = cm.index(next_start)
            image = add_black_chunk(image, 0, cm_index)

            #pixel_img = transform_image(image,feature_extractor)
            

            filename = f'chunk_{i}.jpg'
            image_save_path = os.path.join(test_img_save_dir, filename)
            test_pixels.append(image_save_path)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, image)
            else:
                #print(f"Image '{filename}' already exists. Skipping overwrite.")
                pass

                
        elif next_end in cm[1:5]:
            
            ## Add on to test
            test_indices.append(i)
            test_labels.append(row[target])
            test_depths.append(row['depth'])
            test_sources.append(row['source'])

            ## Take cm from image split to end and keep it. concatenate black square of size start cm[0] - break to front
            image = mpimg.imread(row['chunk'])
            cm_index = cm.index(next_end)
            #print("Check", cm[0:5], i, cm_index)
            image = add_black_chunk(image, cm_index,5)
            #pixel_img = transform_image(image,feature_extractor)

            filename = f'chunk_{i}.jpg'
            image_save_path = os.path.join(test_img_save_dir, filename)
            test_pixels.append(image_save_path)

            if not os.path.exists(image_save_path):
                mpimg.imsave(image_save_path, image)
            else:
                #print(f"Image '{filename}' already exists. Skipping overwrite.")
                pass
        elif state == "train":
            #print("train", i)
            pass
        else:
            print("why are we here?")
            return None

        if cm[1] == next_end:
            ## Only 1 big chunk now; this is irrelavent
            #print(f"resetting {next_start} - {next_end} in {cm}")
            #next_start += 50
            #next_end += 50
            ## Adding on filtered test data for next break //
            #temp_df = test_1cm_df[((test_1cm_df[depth_col] >= next_start) & (test_1cm_df[depth_col] < next_end))].copy()
            #if lake_name != "BOTH":
            #    temp_df['source'] = lake_name
            #else:
            #    temp_df['source'] = "LVID"
            #filtered_test_data =  pd.concat([filtered_test_data, temp_df])
            ## Flipping once were completely past break??
            state = "train"
        elif cm[1] == next_start:
            #print("here")
            state = "test"

        for j in range(len(cm)):
            cm[j] += 1

    return test_indices, test_pixels, test_labels, test_depths, test_sources, filtered_test_data






def create_fold_test_data(lake_name, target, fold, batch_size = 5):
    '''
    Creates fold data with fold specified as test set. Builds test structure and returns paths
    lake_name: lake that image and label data should come from [SVID, LVID, both] (str)
    target: target value for label data (str)
    fold: Specifies which fold as test set. [0,1,2,3,4] (int)
    batch_size: Size of batch in loader. Default is 5 (int)
    '''
    
    lake_name = lake_name.upper()
    if lake_name == "LVID" or lake_name == "SVID":
        test_indices, test_pixels, test_labels, test_depths, test_sources, filtered_test_data = get_fold_test_data(lake_name, target, fold)
        
    elif lake_name == "BOTH": 
        test_indices, test_pixels, test_labels, test_depths, test_sources, filtered_test_data = get_fold_test_data("LVID", target, fold)
        temp_indices, temp_pixels, temp_labels, temp_depths, temp_sources, temp_test_data = get_fold_test_data("SVID", target, fold)
        test_indices += temp_indices
        test_pixels += temp_pixels
        test_labels += temp_labels
        test_depths += temp_depths
        test_sources += temp_sources
        filtered_test_data = pd.concat([filtered_test_data,temp_test_data], axis = 0, ignore_index=True)
    else:
        print("Unimplemented option")
        return "ERROR"

    test_data_folder = f"/Users/willhoff/Desktop/research_23_24/paleoclimate/will_sandbox/MA_data/{lake_name}/test_data/fold_{fold}"
    os.makedirs(test_data_folder, exist_ok=True)
   
    

    # Assuming test_images_df is a DataFrame with indices, depths, sources, and image paths for test images
    # You would need to create this DataFrame similar to how you compiled filtered_test_data
    test_images_df = pd.DataFrame({
        'index': test_indices,
        'depth': test_depths,
        'source': test_sources,
        target: test_labels,
        'image_path': test_pixels  # Assuming this list contains the paths to the saved test images
    })
    # Save filtered test and test image data DataFrame
    filtered_test_data.to_csv(f"{test_data_folder}/test_filtered_{target}_data.csv", index=False)

    test_images_df.to_csv(f"{test_data_folder}/test_images_{target}_data.csv", index=False)

    test_depth_path = f"{test_data_folder}/test_filtered_{target}_data.csv"
    test_img_path = f"{test_data_folder}/test_images_{target}_data.csv"
    return test_depth_path, test_img_path

def get_train_fold_data(lake_name, target, fold, feature_extractor, transform = True):
    '''
    Gets unprepared train data. returns train lists.
    lake_name: lake that image and label data should come from [SVID, LVID, both] (str)
    target: target value for label data (str)
    fold: Specifies which fold as test set. [1,2,3,4,5] (int)
    '''

    data_path = f'MA_data/{lake_name}/data/{lake_name}_dataset.csv'
    df = pd.read_csv(data_path)
    fold_length = int(df.shape[0]/5)
    next_start = fold * fold_length
    next_end = next_start + fold_length

    state = "test" if next_start == 0 else "train"

    cm = [0,1,2,3,4,5]

    # CustomDataset: def __init__(self, indices, pixel_values, labels, depths, sources):
    ## Convert df to CustomDataset
    train_indices, train_pixels, train_labels, train_depths, train_sources = [],[],[],[],[]

    
    for i, row in df.iterrows(): ## for data:
        ## If its in position 5, the opposite data will have a black chunk for the whole thing
        if next_start not in cm[1:5] and next_end not in cm[1:5] and state == "train":
            if not np.isnan(row[target]):
                #print(f'State {state}, Fold {fold}, {i}')
                train_indices.append(i)
                train_labels.append(row[target])
                train_depths.append(row['depth'])
                train_sources.append(row['source'])
                ## Use maybe_add_black_box function that adds a random 1 cm black box into the image with probability 5%
                image = mpimg.imread(row['chunk'])
                image = maybe_add_black_chunk(image)

                pixel_img = transform_image(image,feature_extractor, transform = transform)

                train_pixels.append(pixel_img)

        ## Break in cm --> We also need to increment state somewhere
        elif next_start in cm[1:5]:
            #print(f"break in cm {next_start} of {cm}, state {state}, image {i} \n")
                
            ## Add on other to train
            if not np.isnan(row[target]):
                train_indices.append(i)
                train_labels.append(row[target])
                train_depths.append(row['depth'])
                train_sources.append(row['source'])

                ## Take cm from image split to end and keep it. concatenate black square of size start cm[0] - break to front
                image = mpimg.imread(row['chunk'])
                cm_index = cm.index(next_start)
                image = add_black_chunk(image, cm_index, 5)
                
                pixel_img = transform_image(image,feature_extractor, transform = transform)
            
                train_pixels.append(pixel_img)
            else:
                #print("image nan",cm)
                pass

        elif next_end in cm[1:5]:
            #print(f"break in cm {next_end} of {cm}, state {state}, image {i} \n")
            ## Add on other to train
            if not np.isnan(row[target]):
                train_indices.append(i)
                train_labels.append(row[target])
                train_depths.append(row['depth'])
                train_sources.append(row['source'])

                ## Take cm from image split to end and keep it. concatenate black square of size start cm[0] - break to front
                image = mpimg.imread(row['chunk'])
                cm_index = cm.index(next_end)
                image = add_black_chunk(image, 0, cm_index)
                
                pixel_img = transform_image(image,feature_extractor, transform = transform)

                train_pixels.append(pixel_img)
            else:
                #print("image nan",cm)
                pass
        elif state == "test":
            pass
        else:
            print("why are we here?\n")
            return None


        if cm[1] == next_end:
            #print(f"resetting {next_start} - {next_end} in {cm}")
            # unneeded now that its one big chunk
            #next_start += 50
            #next_end += 50
            state = "train"
        elif cm[1] == next_start:
            state = "test"


        for j in range(len(cm)):
            cm[j] += 1

    return train_indices, train_pixels, train_labels, train_depths, train_sources




def create_fold_train_data(lake_name, target, fold, batch_size = 5, transform = True):
    '''
    Creates fold data with fold specified as test set. returns train Dataloader object.
    lake_name: lake that image and label data should come from [SVID, LVID, both] (str)
    target: target value for label data (str)
    fold_1: Specifies which fold as test set. [1,2,3,4,5] (int)
    fold_2: Only used when lake is "both." Second lake is always SVID. Defaults to fold_1 value if unspecified [1,2,3,4,5] (int)
    batch_size: Size of batch in loader. Default is 5 (int)
    '''
    lake_name = lake_name.upper()
    feature_extractor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
    if lake_name == "BOTH":
        train_indices, train_pixels, train_labels, train_depths, train_sources = get_train_fold_data("LVID",target,fold, feature_extractor, transform = transform)
        temp_indices, temp_pixels, temp_labels, temp_depths, temp_sources = get_train_fold_data("SVID",target,fold, feature_extractor, transform = transform)
        train_indices += temp_indices
        train_pixels += temp_pixels
        train_labels += temp_labels
        train_depths += temp_depths
        train_sources += temp_sources
    else:
        train_indices, train_pixels, train_labels, train_depths, train_sources = get_train_fold_data(lake_name,target,fold, feature_extractor, transform = transform)

    train_dataset = CustomDataset(train_indices, train_pixels, torch.tensor(train_labels, dtype = torch.float32), train_depths, train_sources)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    return loader
        

def create_loader(csv_path, set = "Train", second_csv = None, batch_size = 5):
    '''
    Loads a single DataLoader object from CSV. Returns DataLoader.
    Params:
        csv_path: path to DataFrame CSV (str)
        set: specifies what dataloader you want [Train,Test, NaN, Full] (str)
        second_csv: additional DataFrame csv path (str)
        batch_size: specifies the batch size for the dataloader (int)
    '''
    if set not in ["Test", "Train", np.nan, "Full"]:
        return "options are [\"Test\", \"Train\", np.nan, \"Full\"]"

    df = pd.read_csv(csv_path)
    if set == "Full":
        df = df[(df['set'] == "Train") | (df['set'] == "Test")]
    elif pd.isna(set):
        df = df[pd.isna(df['set'])]
    else:
        df = df[df['set'] == set]

    # Debug: Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty after filtering. Returning None.")
        return None

    feature_extractor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
    df['image_tensor'] = df['chunk'].apply(lambda x: transform_image(x, feature_extractor))
    label_array = np.array(df['label'])
    dataset = CustomDataset(df['index'], df['image_tensor'], torch.tensor(label_array, dtype=torch.float32), df['depth'], df['source'])

    if second_csv is not None:
        df2 = pd.read_csv(second_csv)
        if set == "Full":
            df2 = df2[(df2['set'] == "Train") | (df2['set'] == "Test")]
        elif pd.isna(set):
            df2 = df2[pd.isna(df2['set'])]
        else:
            df2 = df2[df2['set'] == set]

        # Debug: Check if DataFrame is empty
        if df2.empty:
            print("Warning: DataFrame is empty after filtering. Returning None.")
            return None
        ## Missing?? 
        df2['image_tensor'] = df2['chunk'].apply(lambda x: transform_image(x, feature_extractor))
        label_array2 = np.array(df2['label'])
        dataset2 = CustomDataset(df2['index'], df2['image_tensor'], torch.tensor(label_array2, dtype = torch.float32), df2['depth'], df2['source'])
        dataset.concatenate(dataset2, inplace=True)
    
    if set == "Test" or pd.isna(set):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

def create_MA_loader(csv_path, set="Train", second_csv=None, batch_size=5):
    '''
    Loads a single DataLoader object from CSV. Returns DataLoader.
    Params:
        csv_path: path to DataFrame CSV (str)
        set: specifies what DataLoader you want [Train,Test, NaN, Full] (str)
        second_csv: additional DataFrame CSV path (str)
        batch_size: specifies the batch size for the DataLoader (int)
    '''
    if set not in ["Test", "Train", "Full", np.nan]:
        raise ValueError("options are [\"Test\", \"Train\", \"Full\", np.nan]")

    df = pd.read_csv(csv_path)
    # Filter based on the 'set' column
    if set == "Full":
        df = df[((df['set'] == "Train") | (df['set'] == "Test")) & (df['known'] == True)]
    elif pd.isna(set):
        df = df[(df['known'] == False)]
    else:
        df = df[(df['set'] == set) & (df['known'] == True)]

    if df.empty:
        print("Warning: DataFrame is empty after filtering. Returning None.")
        return None

    feature_extractor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
    # Convert image file paths to tensor representations
    df['image_tensor'] = df['chunk'].apply(lambda x: transform_image(x, feature_extractor))
    label_array = np.array(df['label'])
    dataset = CustomDataset(df['index'], df['image_tensor'], torch.tensor(label_array, dtype=torch.float32), df['depth'], df['source'])

    if second_csv is not None:
        df2 = pd.read_csv(second_csv)
        # Apply similar filtering for the second dataset
        if set == "Full":
            df2 = df2[((df2['set'] == "Train") | (df2['set'] == "Test")) & (df2['known'] == True)]
        elif pd.isna(set):
            df2 = df2[(df2['known'] == False)]
        else:
            df2 = df2[(df2['set'] == set) & (df2['known'] == True)]

        if df2.empty:
            print("Warning: Second DataFrame is empty after filtering. Proceeding without it.")
        else:
            df2['image_tensor'] = df2['chunk'].apply(lambda x: transform_image(x, feature_extractor))
            label_array2 = np.array(df2['label'])
            dataset2 = CustomDataset(df2['index'], df2['image_tensor'], torch.tensor(label_array2, dtype=torch.float32), df2['depth'], df2['source'])
            dataset.concatenate(dataset2, inplace=True)

    loader_options = {'batch_size': batch_size, 'shuffle': True if set != "Test" and not pd.isna(set) else False}
    loader = DataLoader(dataset, **loader_options)

    return loader

                        ##### DATALOADERS/TEST/TRAIN DATA ########


def load_data(target, lake = "both", scaled = False, set = "full", percentage = 100, return_labels = False, sediment_width = 5, test_size = .2, random = True):
    '''
    Loads a given DataLoader Object. 
    Params:
        target: %TOC or MBT. specifies which variable to keep as the target variable
        lake: Specifies which lake to use (both, lvid, svid)
        scaled: bool specifies whether object should be scaled
        set: Possible values of full, train, test based on whether the caller wants both train and test, only train or only test for a given dataset
        percentage: percentage of total data to use when making initial split. Make sure its same 50% each time
        sediment_width: width of image slices. Default is 5 cm        
    set:
    '''

    ### CHANGE LOCAL DATA DIR HERE
        ## Probably should make a referential way to do this
    local_data_file = "/Users/willhoff/Desktop/thesis_2024/data/tabular/SVID_TOC.csv"
    if lake == "both":

        full_SVID = []
        full_LVID = [] 

        ## Images
        load_and_display_images('img_data/SVID',full_SVID, "SVID")
        ## reverse = False because 1B-7B seems darkest so would make sense for it to be on the bottom
        full_SVID.sort(key=sortimg,reverse=False)

        load_and_display_images('img_data/LVID',full_LVID, "LVID")
        ## reverse = False because 1B-7B seems darkest so would make sense for it to be on the bottom
        full_LVID.sort(key=sortimg,reverse=False)

        cropped_LVID = crop_lvid(full_LVID)
        cropped_SVID = crop_svid(full_SVID)

        if target == "MBT":
            DATA_DIR = "../data/"
            with open(os.path.join(DATA_DIR, "SVID_brGDGTs.csv"), "r") as inf:
                svid_b = pd.read_csv(inf)
            lvid_b = pd.read_excel('../data/HEID_LVID_brGDGTs.xlsx','LVID brGDGTs')
            lvid_b.columns = [clean_column_name(col) for col in lvid_b.columns]
            lvid_b.rename(columns={'mbt\'-5me': "MBT"}, errors="raise", inplace=True) 

            ## Figure out how to scale values together
            lvid_pixel_values_tensor, lvid_labels_tensor, lvid_scaler,Ldepths, Lsource, Lindices = attach_labels(target, lvid_b, full_LVID, "cum_depth", "LVID", scaled=scaled, sediment_width = sediment_width)
            svid_pixel_values_tensor, svid_labels_tensor, svid_scaler,Sdepths, Ssource, Sindices = attach_labels(target, svid_b, full_SVID, "Sediment_Depth", "SVID", scaled=scaled, sediment_width = sediment_width)

            ## If sequential: maybe move to create dataset somehow??
            if not random:
                lvid_dataset = CustomDataset(Lindices, lvid_pixel_values_tensor, lvid_labels_tensor, Ldepths, Lsource)
                svid_dataset = CustomDataset(Sindices,svid_pixel_values_tensor, svid_labels_tensor, Sdepths, Ssource)
                Lsplit_idx = int((1-test_size) * len(lvid_labels_tensor))
                Ssplit_idx = int((1-test_size) * len(svid_labels_tensor))
                Ltrain_indices = list(range(Lsplit_idx))
                Strain_indices = list(range(Ssplit_idx))
                Lval_indices = list(range(Lsplit_idx, len(lvid_labels_tensor)))
                Sval_indices = list(range(Ssplit_idx, len(svid_labels_tensor)))

                # Subset for train and validation
                Ltrain_dataset = Subset(lvid_dataset, Ltrain_indices)
                Strain_dataset = Subset(svid_dataset, Strain_indices)
                Lval_dataset = Subset(lvid_dataset, Lval_indices)
                Sval_dataset = Subset(svid_dataset, Sval_indices)

                # Concatenate training and validation subsets from both lakes
                full_train_dataset = ConcatDataset([Ltrain_dataset, Strain_dataset])
                full_val_dataset = ConcatDataset([Lval_dataset, Sval_dataset])

                # Create a dataloader for both the training and validation sets
                train_loader = DataLoader(full_train_dataset, batch_size=5, shuffle=True)
                val_loader = DataLoader(full_val_dataset, batch_size=5, shuffle=False)

                return train_loader, val_loader, None  
                
            img_tensor = torch.cat((lvid_labels_tensor,svid_pixel_values_tensor))
            labels_tensor = torch.cat((lvid_labels_tensor,svid_labels_tensor))
            
            ## Correct way to append??
            depths = np.append(Ldepths,Sdepths)

            sources = np.append(Lsource, Ssource)

            indices = np.append(Lindices, Sindices)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(img_tensor, labels_tensor,depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler,labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"
        elif target == "%TOC":
            svid_o = pd.read_csv(local_data_file)
            lvid_o = pd.read_excel('../data/LVID_bulk_geochem.xlsx')
            # Apply the function to each column name: LVID lower case
            lvid_o.columns = [clean_column_name(col) for col in lvid_o.columns]
            
            lvid_pixel_values_tensor, lvid_labels_tensor, lvid_scaler,Ldepths, Lsource, Lindices  = attach_labels("%toc", lvid_o, full_LVID, "cum_depth", "LVID", scaled=scaled, sediment_width = sediment_width)
            svid_pixel_values_tensor, svid_labels_tensor, svid_scaler, Sdepths, Ssource, Sindices  = attach_labels(target, svid_o, full_SVID, "Sediment_Depth", "SVID", scaled=scaled, sediment_width = sediment_width)

            ## If sequential:
            if not random:
                lvid_dataset = CustomDataset(Lindices,lvid_pixel_values_tensor, lvid_labels_tensor, Ldepths, Lsource)
                svid_dataset = CustomDataset(Sindices,svid_pixel_values_tensor, svid_labels_tensor, Sdepths, Ssource)
                Lsplit_idx = int((1-test_size) * len(lvid_labels_tensor))
                Ssplit_idx = int((1-test_size) * len(svid_labels_tensor))
                Ltrain_indices = list(range(Lsplit_idx))
                Strain_indices = list(range(Ssplit_idx))
                Lval_indices = list(range(Lsplit_idx, len(lvid_labels_tensor)))
                Sval_indices = list(range(Ssplit_idx, len(svid_labels_tensor)))

                # Subset for train and validation
                Ltrain_dataset = Subset(lvid_dataset, Ltrain_indices)
                Strain_dataset = Subset(svid_dataset, Strain_indices)
                Lval_dataset = Subset(lvid_dataset, Lval_indices)
                Sval_dataset = Subset(svid_dataset, Sval_indices)

                # Concatenate training and validation subsets from both lakes
                full_train_dataset = ConcatDataset([Ltrain_dataset, Strain_dataset])
                full_val_dataset = ConcatDataset([Lval_dataset, Sval_dataset])

                # Create a dataloader for both the training and validation sets
                train_loader = DataLoader(full_train_dataset, batch_size=5, shuffle=True)
                val_loader = DataLoader(full_val_dataset, batch_size=5, shuffle=False)

                return train_loader, val_loader, None  



            img_tensor = torch.cat((lvid_pixel_values_tensor,svid_pixel_values_tensor))
            labels_tensor = torch.cat((lvid_labels_tensor,svid_labels_tensor))

            ## Correct way to append??
            depths = np.append(Ldepths,Sdepths)

            sources = np.append(Lsource, Ssource)
            indices = np.append(Lindices, Sindices)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(img_tensor, labels_tensor,depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler,labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"

        else:
            return "target param options: MBT, %TOC"
    elif lake == "lvid":

        full_LVID = [] 

        load_and_display_images('img_data/LVID',full_LVID, "LVID")
        ## reverse = False because 1B-7B seems darkest so would make sense for it to be on the bottom
        full_LVID.sort(key=sortimg,reverse=False)

        cropped_LVID = crop_lvid(full_LVID)
        

        if target == "MBT":
            lvid_b = pd.read_excel('../data/HEID_LVID_brGDGTs.xlsx','LVID brGDGTs')
            lvid_b.columns = [clean_column_name(col) for col in lvid_b.columns]
            lvid_b.rename(columns={'mbt\'-5me': "MBT"}, errors="raise", inplace=True) 

            ## Figure out how to scale values together
            lvid_pixel_values_tensor, lvid_labels_tensor, lvid_scaler,depths,sources, indices = attach_labels(target, lvid_b, full_LVID, "cum_depth", "LVID", scaled=scaled, sediment_width = sediment_width)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(lvid_pixel_values_tensor, lvid_labels_tensor,depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler,lvid_labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"
        elif target == "%TOC":
            lvid_o = pd.read_excel('../data/LVID_bulk_geochem.xlsx')
            # Apply the function to each column name: LVID lower case
            lvid_o.columns = [clean_column_name(col) for col in lvid_o.columns]

            lvid_pixel_values_tensor, lvid_labels_tensor, lvid_scaler,depths,sources, indices = attach_labels("%toc", lvid_o, full_LVID, "cum_depth", "LVID",scaled=scaled, sediment_width = sediment_width)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(lvid_pixel_values_tensor, lvid_labels_tensor,depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler,lvid_labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"
        else:
            return "target param options: MBT, %TOC"
    elif lake == "svid":

        full_SVID = []

        ## Images --> change this dir for different local structure
        image_dir = "/Users/willhoff/Desktop/thesis_2024/data/img_data/SVID"
        load_and_display_images(image_dir,  full_SVID, "SVID")
        ## reverse = False because 1B-7B seems darkest so would make sense for it to be on the bottom
        full_SVID.sort(key=sortimg,reverse=False)

        cropped_SVID = crop_svid(full_SVID)

        if target == "MBT":
            DATA_DIR = "../data/"
            with open(os.path.join(DATA_DIR, "SVID_brGDGTs.csv"), "r") as inf:
                svid_b = pd.read_csv(inf)

            svid_pixel_values_tensor, svid_labels_tensor, svid_scaler,depths, sources, indices = attach_labels(target, svid_b, full_SVID, "Sediment_Depth", "SVID", scaled=scaled, sediment_width = sediment_width)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(svid_pixel_values_tensor, svid_labels_tensor,depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler,svid_labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"

        elif target == "%TOC":
            svid_o = pd.read_csv(local_data_file)

            svid_pixel_values_tensor, svid_labels_tensor, svid_scaler,depths,sources, indices = attach_labels(target, svid_o, full_SVID, "Sediment_Depth", "SVID", scaled=scaled, sediment_width = sediment_width)

            ## train_loader, val_loader, scaler 
            train_loader, val_loader, scaler  = create_dataset(svid_pixel_values_tensor, svid_labels_tensor, depths,sources, indices, random=random)
            if set == "full":
                if return_labels:
                    return train_loader, val_loader, scaler, svid_labels_tensor
                else:
                    return train_loader, val_loader, scaler
            elif set == "test":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return val_loader, scaler
            elif set == "train":
                if return_labels:
                    return "Only allowed when set = full"
                else:
                    return train_loader, scaler
            else:
                return "set param options: full, test,train"


        else:
            return "target param options: MBT, %TOC"
    else:
        return "lake param options: lvid, svid, both"
    
def create_dataset(images, labels, depths, sources, indices, scaled=False, test_size = 0.2, random =True):
    
    ## Previously scaling was being done as scaled=True. Fixed now 1-24-24
    scaler = None
    if scaled:
         # Ensure labels are a numpy array
        labels = np.array(labels)

        # Reshape labels to 2D for scaling
        labels_2d = labels.reshape(-1, 1)

        scaler = StandardScaler()
        scaled_labels = scaler.fit_transform(labels_2d)

        # Reshape the scaled labels back to 1D
        labels = scaled_labels.flatten()
    
    ## Use VIT --> moved back out
    # images = transform_images(images)

    # Create the custom dataset
    dataset = CustomDataset(indices, images, labels, depths, sources)

    # Split the dataset into training and validation sets

    if random:
        train_indices, val_indices = train_test_split(range(len(labels)), test_size=test_size, random_state=2)
    else:
       split_idx = int((1-test_size) * len(labels))
       train_indices = list(range(split_idx))
       val_indices = list(range(split_idx, len(labels)))
    
    # Subset for train and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create a dataloader for both the training and validation sets
        ## 5 seems like a fine batch size
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    return train_loader, val_loader, scaler