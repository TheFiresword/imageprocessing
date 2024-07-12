import requests
import time
from pathlib import Path
import os, glob, re
import cv2
from typing import List
import micasense.capture as capture
import micasense.imageutils as imageutils
import skimage
import logging
import numpy as np
from skimage.transform import warp,matrix_transform,resize,FundamentalMatrixTransform,estimate_transform,ProjectiveTransform



def check_exst_warp_matrices(warp_matrices_filename: str):
    if Path('./' + warp_matrices_filename).is_file():
        print("Found existing warp matrices for camera")
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices: 
            loaded_warp_matrices.append(matrix.astype('float32'))
        print("Warp matrices successfully loaded.")
        warp_matrices = loaded_warp_matrices
    else:
        print("No existing warp matrices found.")
        warp_matrices = False
    return warp_matrices


def save_warp_matrices(warp_matrices : list, warp_matrices_filename : str, rewrite : bool = True):
    working_wm = warp_matrices
    if not Path('./' + warp_matrices_filename).is_file() or rewrite:
        temp_matrices = []
        for x in working_wm:
            if isinstance(x, np.ndarray):
                temp_matrices.append(x)
            if isinstance(x, skimage.transform._geometric.ProjectiveTransform):
                temp_matrices.append(x.params)
        np.save(warp_matrices_filename, np.array(temp_matrices, dtype=object), allow_pickle=True)
        print("Saved to", Path('./' + warp_matrices_filename).resolve())
    else:
        print("Matrices already exist at",Path('./' + warp_matrices_filename).resolve())
    return


def save_aligned_images(im_aligned, thecapture : capture.Capture, output_dir : str, stacked : bool = True):
    os.makedirs(output_dir, exist_ok=True)
    band_images_names = [Path(img.path).name  for img in thecapture.images]
    
    if stacked:
        # save the 5 bands as a single stacked tif file in RGB color
        rgb_band_indices = [thecapture.band_names_lower().index('red'),
                        thecapture.band_names_lower().index('green'),
                        thecapture.band_names_lower().index('blue')]

        # Create normalized stacks for viewing
        im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],im_aligned.shape[2]), dtype=np.float32)
        im_min = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
        im_max = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values

        # for rgb true color, we use the same min and max scaling across the 3 bands to 
        # maintain the "white balance" of the calibrated image
        for i in rgb_band_indices:
            im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i], im_min, im_max)
        rgb = im_display[:,:,rgb_band_indices]
        stacked_rgb_img_name = f"IMG_{thecapture.uuid}.tif"
        cv2.imwrite(os.path.join(output_dir, stacked_rgb_img_name), (rgb * (2**8 - 1)).astype(np.uint8))
    else:
        for band_name in thecapture.band_names_lower():
            band_index = thecapture.band_names_lower().index(band_name)
            band_image = im_aligned[:, :, band_index]            
            filename = band_images_names[band_index]
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, (band_image * (2**16 - 1)).astype(np.uint8))  #  astype(np.uint16)  doesn't show the images normally
    return


# Fonction principale
def realign_images(set_root_path: str = "", 
                   panels_ids : List[str] = [],
                   regenerate_matrices : bool = True, 
                   save_as_stack : bool = True,
                   output_dir_name : str="aligned",
                   pyramid_levels: int = 2,
                   max_alignment_iterations: int = 50):
    # pyramid_levels:: for images with RigRelatives, setting this to 0 or 1 may improve alignment
    images_path = Path(set_root_path)
    panels_ids = panels_ids or []
    if not images_path.exists() or not images_path.is_dir():
        print(f"Le répertoire {images_path} n'existe pas ou n'est pas un répertoire.")

    captures = {}
    panels = {}
    for image_path in images_path.glob('IMG_*.tif'):
        match = re.match(r'IMG_(\d+)_\d\.tif', image_path.name)
        if match:
            capture_number = match.group(1)
            if capture_number in panels_ids:
                # c'est une image du panneau
                if capture_number not in panels:
                    panels[capture_number] = []
                panels[capture_number].append(image_path.as_posix())
            else:
                # c'est une image normale
                if capture_number not in captures:
                    captures[capture_number] = []
                captures[capture_number].append(image_path.as_posix())

    valid_captures = {k: v for k, v in captures.items() if len(v) == 5} if captures != {} else None
    valid_panels = {k: v for k, v in panels.items() if len(v) == 5} if panels != {} else None
    
    if valid_panels:
        for capture_number, imageNames in valid_panels.items():
            panel_capture = capture.Capture.from_filelist(imageNames)
    
    if valid_captures:
        for capture_number, imageNames in valid_captures.items():
            thecapture = capture.Capture.from_filelist(imageNames)
            if valid_panels: img_type = "reflectance"
            elif thecapture.dls_present(): img_type='reflectance'
            else: img_type='radiance'
            
            cam_serial = thecapture.camera_serial
            warp_matrices_filename = str(cam_serial) + "_warp_matrices_opencv.npy"
            warp_matrices = check_exst_warp_matrices(warp_matrices_filename)
            if warp_matrices is False or regenerate_matrices is True:
                st = time.time()
                match_index = 1
                warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
                print(f"Aligning images of capture {capture_number}")
                try:
                    warp_matrices, alignment_pairs = imageutils.align_capture(thecapture,
                                                                ref_index = match_index,
                                                                max_iterations = max_alignment_iterations,
                                                                warp_mode = warp_mode,
                                                                pyramid_levels = pyramid_levels)                    
                except Exception:
                    logging.warning(f"The alignment algorithm failed for capture {capture_number}\n Skipping this capture")  
                
                else:
                    save_warp_matrices(warp_matrices, warp_matrices_filename=warp_matrices_filename, rewrite=regenerate_matrices)
                    cropped_dimensions, edges = imageutils.find_crop_bounds(thecapture, warp_matrices, warp_mode=warp_mode, reference_band=match_index)
                    im_aligned = thecapture.create_aligned_capture(warp_matrices=warp_matrices, motion_type=warp_mode, img_type=img_type)
                    try:
                        save_aligned_images(im_aligned, thecapture, set_root_path+"/"+output_dir_name, stacked=save_as_stack)
                        #thecapture.save_capture_as_stack(thecapture.uuid+"-noPanels.tif", sort_by_wavelength=True)
                    except Exception as e:
                        logging.warning(f"Couldn't save images of capture {thecapture.uuid}")
                        print(e)
                    
                    print(f"Finished Aligning after {int(time.time() - st)} seconds")           
            else:
                print("Using existing warp matrices...")
                im_aligned = thecapture.create_aligned_capture(warp_matrices=warp_matrices, motion_type=warp_mode, img_type=img_type)
                try:
                    save_aligned_images(im_aligned, thecapture, set_root_path+"/"+output_dir_name, stacked=save_as_stack)
                except Exception:
                    logging.warning(f"Couldn't save images of capture {thecapture.uuid}")
                print(f"Finished Aligning after {int(time.time() - st)} seconds")        
            


# Exemple d'utilisation
realign_images(set_root_path = "./data/TEDDY/0001/000", panels_ids = [],
    regenerate_matrices = True, save_as_stack = True, output_dir_name = "aligned", pyramid_levels=3, 
    max_alignment_iterations=50)