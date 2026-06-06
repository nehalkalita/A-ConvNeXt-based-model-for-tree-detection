import os
import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from scipy.ndimage import label

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# CPU single-core enforcement
os.environ["OMP_NUM_THREADS"]        = "2"
os.environ["MKL_NUM_THREADS"]        = "2"
os.environ["OPENBLAS_NUM_THREADS"]   = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"]    = "2"

import onnxruntime as ort


#  Configure paths

# Path to the exported .onnx file
# The matching .onnx.data file must sit in the same folder
ONNX_MODEL = r"convunext\convunext_cls_best_gis.onnx"

# Single whole image file (wi)  OR  a folder of whole image (wi) files to process
INPUT = r"convunext\test_wi\4.png"

# Folder where all output files are saved
# A separate subfolder is created per input whole image (wi)
OUTPUT_DIR = r"convunext\test_output_whole"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alpha transparency for the class colour overlays on the composite image
# Range 0 (invisible) – 255 (fully opaque).  180 as specified.
OVERLAY_ALPHA = 180

# If True, prints the (row, col) patch index for every patch processed.
# Useful for debugging but noisy for large images — set False for normal use.
VERBOSE_PATCHES = False

dim = 400 # length of each side of the dimension
img_divisions = 3 # no. of divisions of 'dim'
k_nbr = 1 #8 # value for k nearest neighbour
min_accepted_a0 = 0 #0.005 # min area for individual trees
min_accepted_a1 = 0 #0.0005 # min area for cluster of trees


#  Fixed Constants

MEAN           = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD            = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


#  Device Selection

def select_providers() -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        print("  Device : CUDA GPU (onnxruntime-gpu)")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print("  Device : CPU (single-core mode)")
        return ["CPUExecutionProvider"]

# FUNCTION TO SEGMENT PIXELS
def seg_Color(count_pxl, hgt, wdt):
    all_pixels = [[] for i in range(hgt)] # hgt (y) no. of lists with wdtH1 (x) no. of nested lists
    for i in range(hgt):
        all_pixels[i] = [[] for j in range(wdt)]
    
    # detect segments in each row (vertical thickness of each segment is 1 pixel)
    for i1 in range(len(count_pxl)):
        all_pixels[count_pxl[i1][1]][count_pxl[i1][0]] = [1]

    # Convert your image format to a binary numpy array
    image_np = np.array([[1 if cell else 0 for cell in row] for row in all_pixels])

    # 4-connectivity = cross-shaped structuring element (no diagonals)
    struct = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])

    cluster_map, num_clusters = label(image_np, structure=struct)

    # Collect pixel locations of each cluster
    segs = [[] for i in range(num_clusters)]
    for i1 in range(len(cluster_map)):
        for i2 in range(len(cluster_map[i1])):
            if cluster_map[i1][i2] > 0:
                segs[cluster_map[i1][i2] - 1].append([i2, i1])
    return segs

def short_dist(point1, point2): # shortest distance
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    
    squared_diff_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    return math.sqrt(squared_diff_sum)


#  Load ONNX Model

def load_session(providers: list):
    """
    Returns (session, img_size, num_classes, input_name, output_name).
    img_size and num_classes come from the ONNX graph shape metadata.
    """
    if not Path(ONNX_MODEL).exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads     = 2
    sess_opts.inter_op_num_threads     = 2
    sess_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        ONNX_MODEL,
        sess_options = sess_opts,
        providers    = providers,
    )

    inp        = session.get_inputs()[0]
    input_name = inp.name
    img_size   = int(inp.shape[2])        # H dimension (== W for this model)

    out         = session.get_outputs()[0]
    output_name = out.name
    num_classes = int(out.shape[1])       # C dimension

    print(f"\n  ONNX model  : {ONNX_MODEL}")
    print(f"  Classes     : {num_classes}")
    print(f"  Patch size  : {img_size} × {img_size}")
    print(f"  Stride      : {img_size // 3} px  (1/3 of patch size)")
    print(f"  Input node  : '{input_name}'  shape {list(inp.shape)}")
    print(f"  Output node : '{output_name}' shape {list(out.shape)}")

    return session, img_size, num_classes, input_name, output_name


#  RGBA-safe Image Loader

def load_image_rgb(path: str) -> np.ndarray:
    """Returns a guaranteed (H, W, 3) uint8 RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


#  Patch Preprocessing

def preprocess_patch(patch: np.ndarray, img_size: int) -> np.ndarray:
    """
    patch  : (img_size, img_size, 3) uint8 RGB
    Returns: (1, 3, img_size, img_size) float32 — ImageNet normalised
    """
    if patch.shape[:2] != (img_size, img_size):
        patch = cv2.resize(patch, (img_size, img_size),
                           interpolation=cv2.INTER_LINEAR)
    img_f = patch.astype(np.float32) / 255.0
    img_f = (img_f - MEAN) / STD                          # (H, W, 3)
    img_f = img_f.transpose(2, 0, 1)[np.newaxis, ...]    # (1, 3, H, W)
    return img_f.astype(np.float32)


#  Class Colours & Names

def make_class_colors(num_classes: int, alpha: int = 180) -> np.ndarray:
    """
    Returns (num_classes, 4) RGBA uint8.
        Class 0          -> transparent
        Classes 1 … n-1  -> distinct tab10 colours at given alpha
    """
    #cmap   = plt.cm.get_cmap("tab10")
    cmap   = plt.get_cmap("tab10")
    colors = [[0, 0, 0, 0]]
    for i in range(1, num_classes):
        r, g, b, _ = cmap((i - 1) % 10)
        colors.append([int(r * 255), int(g * 255), int(b * 255), alpha])
    return np.array(colors, dtype=np.uint8)


def make_class_names(num_classes: int) -> list:
    return ["Background"] + [f"Class {c}" for c in range(1, num_classes)]


#  Sliding Window Inference

def infer_wi(session:      ort.InferenceSession,
              wi:          np.ndarray,
              img_size:     int,
              num_classes:  int,
              input_name:   str,
              output_name:  str) -> np.ndarray:
    """
    Slides a window of img_size x img_size across the wi with
    stride = img_size // 3.

    Overlapping patches contribute to the same output pixels.
    Their softmax probabilities are accumulated and divided by the
    per-pixel visit count to produce an average probability map.
    The argmax of the averaged map gives the final label per pixel.

    Args:
        wi : (m1, m2, 3) uint8 RGB whole image
    Returns:
        pred_map : (m1, m2) int64  — per-pixel class label
    """
    m1, m2    = wi.shape[:2]
    stride    = img_size // 3

    #  Pad the wi so every possible top-left position yields a full patch 
    # We pad on the right and bottom only, preserving top-left coordinates.
    pad_h = (math.ceil((m1 - img_size) / stride) * stride
             + img_size - m1) if m1 > img_size else (img_size - m1)
    pad_w = (math.ceil((m2 - img_size) / stride) * stride
             + img_size - m2) if m2 > img_size else (img_size - m2)

    # Reflect padding preserves local texture at the edges
    wi_padded = np.pad(
        wi,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )
    ph, pw = wi_padded.shape[:2]

    #  Accumulation buffers (float64 for precision when many patches overlap)
    prob_acc  = np.zeros((num_classes, ph, pw), dtype=np.float64)
    count_map = np.zeros((ph, pw),              dtype=np.float64)

    #  Build list of all (row, col) top-left patch positions 
    rows = list(range(0, ph - img_size + 1, stride))
    cols = list(range(0, pw - img_size + 1, stride))

    total_patches = len(rows) * len(cols)

    # Progress display
    if TQDM_AVAILABLE:
        patch_iter = tqdm(
            [(r, c) for r in rows for c in cols],
            desc="  Patches",
            unit="patch",
            ncols=72,
        )
    else:
        patch_iter = [(r, c) for r in rows for c in cols]
        print(f"  Processing {total_patches} patches "
              f"({len(rows)} rows x {len(cols)} cols) …")

    processed = 0
    for (row, col) in patch_iter:
        patch = wi_padded[row:row + img_size, col:col + img_size]

        inp   = preprocess_patch(patch, img_size)
        logit = session.run([output_name], {input_name: inp})[0][0]
        # logit shape: (num_classes, img_size, img_size)

        # Softmax (numerically stable)
        logit -= logit.max(axis=0, keepdims=True)
        exp    = np.exp(logit)
        probs  = exp / exp.sum(axis=0, keepdims=True)

        # Accumulate into the padded buffers
        prob_acc[:, row:row + img_size, col:col + img_size]  += probs
        count_map[   row:row + img_size, col:col + img_size] += 1.0

        processed += 1
        if VERBOSE_PATCHES and not TQDM_AVAILABLE:
            print(f"    patch ({row:>5}, {col:>5})  "
                  f"{processed}/{total_patches}")

    if not TQDM_AVAILABLE:
        print(f"  All {total_patches} patches processed.")

    #  Average accumulated probabilities 
    # count_map is > 0 everywhere because padding ensures full coverage
    prob_avg = prob_acc / np.maximum(count_map, 1e-8)

    #  Crop back to original wi dimensions 
    prob_avg = prob_avg[:, :m1, :m2]    # (num_classes, m1, m2)

    #  Argmax -> label map 
    pred_map = prob_avg.argmax(axis=0).astype(np.int64)   # (m1, m2)

    return pred_map


#  Generate Output Images

def build_binary_masks(pred_map:    np.ndarray,
                       num_classes: int) -> list:
    """
    Returns a list of n-1 binary mask arrays, one per foreground class.
        index 0 -> Class 1 mask  (white where pred==1, black elsewhere)
        index 1 -> Class 2 mask  (white where pred==2, black elsewhere)
        …
    Each array is (m1, m2) uint8  — values 0 or 255.
    """
    masks = []
    for c in range(1, num_classes):
        binary = np.where(pred_map == c, 255, 0).astype(np.uint8)
        masks.append(binary)
    return masks


def build_composite(wi:          np.ndarray,
                    pred_map:     np.ndarray,
                    class_colors: np.ndarray) -> np.ndarray:
    """
    Overlays class colour masks onto the original wi.
        - Background (class 0) is transparent -> original pixels show through
        - Each foreground class gets its tab10 colour at OVERLAY_ALPHA
    Returns (m1, m2, 3) uint8 RGB.
    """
    rgba  = class_colors[pred_map]                 # (m1, m2, 4) RGBA
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

    blended = (wi.astype(np.float32) * (1.0 - alpha) +
               rgba[:, :, :3].astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


#  Save Outputs

def save_outputs(wi:          np.ndarray,
                 pred_map:     np.ndarray,
                 wi_name:     str,
                 out_dir:      Path,
                 num_classes:  int,
                 class_colors: np.ndarray,
                 class_names:  list):
    """
    Saves n output files into out_dir:
        {stem}_class_{c}_mask.png   for c in 1 … n-1  (binary, greyscale)
        {stem}_composite.png                            (colour overlay on wi)

    Also prints per-class pixel statistics to terminal.
    """
    stem         = Path(wi_name).stem
    binary_masks = build_binary_masks(pred_map, num_classes)
    img_height, img_width = wi.shape[:2]

    all_pixels, seg_vld, seg_oline, valid_rows, valid_columns = [], [], [], [], []

    # Collect positions of valid pixels (non-background)
    for idx, (c, mask) in enumerate(zip(range(1, num_classes), binary_masks)):
        count_vld = []
        valid_rows.append([])
        valid_rows[-1] = [img_height, 0] # min, max
        for i1 in range(len(mask)):
            flag1 = 0
            for i2 in range(len(mask[i1])):
                if mask[i1][i2] == 0:
                    pass
                else:
                    flag1 = 1
                    count_vld.append([i2, i1])
            if flag1 == 1:
                valid_rows[-1].append(i1)
                if i1 < valid_rows[-1][0]:
                    valid_rows[-1][0] = i1
                if i1 > valid_rows[-1][1]:
                    valid_rows[-1][1] = i1
        
        valid_columns.append([])
        valid_columns[-1] = [img_width, 0] # min, max
        for i1 in range(valid_rows[-1][0], valid_rows[-1][1] + 1):
            for i2 in range(len(mask[i1])):
                if mask[i1][i2] == 0:
                    pass
                else:
                    if i2 < valid_columns[-1][0]:
                        valid_columns[-1][0] = i2
                    break
            for i2 in reversed(range(len(mask[i1]))):
                if mask[i1][i2] == 0:
                    pass
                else:
                    if i2 > valid_columns[-1][1]:
                        valid_columns[-1][1] = i2
                    break

        # segment pixels
        print(f'Class_{c} Valid rows range: {valid_rows[-1][0]}, {valid_rows[-1][1]}')
        print(f'Class_{c} Valid column range: {valid_columns[-1][0]}, {valid_columns[-1][1]}')
        seg_vld.append([])
        all_pixels.append([])
        if len(count_vld) > 0:
            seg_vld[-1] = seg_Color(count_vld, img_height, img_width)
            count_vld = []
        print('Completed seg_Color()')
        
        all_pixels[-1] = [[] for i in range(img_height)] # hgtH1 (y) no. of lists with wdtH1 (x) no. of nested lists
        for i in range(img_height):
            all_pixels[-1][i] = [-1 for j in range(img_width)]
        
        for j1 in range(len(seg_vld[-1])):
            for j2 in range(len(seg_vld[-1][j1])):
                all_pixels[-1][seg_vld[-1][j1][j2][1]][seg_vld[-1][j1][j2][0]] = j1
        
        # FIND SEGMENT OUTLINE              (NOTE: can create errors while testing some images)
        seg_oline.append([])
        seg_oline[-1] = [[] for i in range(len(seg_vld[-1]))]
        #seg_oline[-1] = [[] for i in range(seg_vld[-1])]
        
        if valid_columns[-1][1] >= valid_columns[-1][0]:
            # First row
            if all_pixels[-1][valid_rows[-1][0]][valid_columns[-1][0]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[-1][0]][valid_columns[-1][0]]].append([valid_columns[-1][0], valid_rows[-1][0]])
                
            for i2 in range(valid_columns[-1][0] + 1, valid_columns[-1][1]): # valid_rows[0]
                if all_pixels[-1][valid_rows[-1][0]][i2] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][valid_rows[-1][0]][i2]].append([i2, valid_rows[-1][0]])

            if all_pixels[-1][valid_rows[-1][0]][valid_columns[-1][1]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[-1][0]][valid_columns[-1][1]]].append([valid_columns[-1][1], valid_rows[-1][0]])
            
            # Rows in between first and last
            for i1 in range(valid_rows[-1][0] + 1, valid_rows[-1][1]):
                if all_pixels[-1][i1][valid_columns[-1][0]] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][i1][valid_columns[-1][0]]].append([valid_columns[-1][0], i1])

                for i2 in range(valid_columns[-1][0] + 1, valid_columns[-1][1]):
                    if all_pixels[-1][i1][i2] > -1: # valid pixel (not background)
                        temp_list = [0, 0, 0, 0] # left, right, top, bottom
                        if all_pixels[-1][i1][i2 - 1] > -1:
                            temp_list[0] = 1
                        if all_pixels[-1][i1][i2 + 1] > -1:
                            temp_list[1] = 1
                        if all_pixels[-1][i1 - 1][i2] > -1:
                            temp_list[2] = 1
                        if all_pixels[-1][i1 + 1][i2] > -1:
                            temp_list[3] = 1
                        if temp_list[0] == 1 and temp_list[1] == 1 and temp_list[2] == 1 and temp_list[3] == 1:
                            pass
                        else:
                            seg_oline[-1][all_pixels[-1][i1][i2]].append([i2, i1])

                if all_pixels[-1][i1][valid_columns[-1][1]] > -1: # valid pixel (not background)
                    #print(all_pixels[-1][i1][valid_columns[-1][1]], len(seg_oline[-1]) - 1)
                    seg_oline[-1][all_pixels[-1][i1][valid_columns[-1][1]]].append([valid_columns[-1][1], i1])
            
            # Last row
            if all_pixels[-1][valid_rows[-1][1]][valid_columns[-1][0]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[-1][1]][valid_columns[-1][0]]].append([valid_columns[-1][0], valid_rows[-1][1]])
                
            for i2 in range(valid_columns[-1][0] + 1, valid_columns[-1][1]): # valid_rows[0]
                if all_pixels[-1][valid_rows[-1][1]][i2] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][valid_rows[-1][1]][i2]].append([i2, valid_rows[-1][1]])

            if all_pixels[-1][valid_rows[-1][1]][valid_columns[-1][1]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[-1][1]][valid_columns[-1][1]]].append([valid_columns[-1][1], valid_rows[-1][1]])
        
        # STORE all_pixels, seg_vld, seg_oline, valid_rows AND valid_columns
    
    # Find Class 2 in Class 1
    temp_list = [] # idx_class_1, idx_class_2

    for k in range(2):
        for i0 in range(len(seg_oline[0])):
            for i1 in seg_oline[0][i0]:
                if (i1[0] - 1 > -1) and all_pixels[1][i1[1]][i1[0] - 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1]][i1[0] - 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1]][i1[0] - 1]])
                        break
                if (i1[0] + 1 < len(all_pixels[1][i1[1]])) and all_pixels[1][i1[1]][i1[0] + 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1]][i1[0] + 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1]][i1[0] + 1]])
                        break
                if (i1[1] - 1 > -1) and all_pixels[1][i1[1] - 1][i1[0]] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] - 1][i1[0]]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] - 1][i1[0]]])
                        break
                if (i1[1] + 1 < len(all_pixels[1])) and all_pixels[1][i1[1] + 1][i1[0]] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] + 1][i1[0]]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] + 1][i1[0]]])
                        break
                if (i1[0] - 1 > -1) and (i1[1] - 1 > -1) and all_pixels[1][i1[1] - 1][i1[0] - 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] - 1][i1[0] - 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] - 1][i1[0] - 1]])
                        break
                if (i1[0] + 1 < len(all_pixels[1][i1[1] - 1])) and (i1[1] - 1 > -1) and all_pixels[1][i1[1] - 1][i1[0] + 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] - 1][i1[0] + 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] - 1][i1[0] + 1]])
                        break
                if (i1[0] - 1 > -1) and (i1[1] + 1 < len(all_pixels[1])) and all_pixels[1][i1[1] + 1][i1[0] - 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] + 1][i1[0] - 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] + 1][i1[0] - 1]])
                        break
                if (i1[1] + 1 < len(all_pixels[1])) and (i1[0] + 1 < len(all_pixels[1][i1[1] + 1])) and all_pixels[1][i1[1] + 1][i1[0] + 1] != -1:
                    if temp_list.__contains__([i0, all_pixels[1][i1[1] + 1][i1[0] + 1]]) == False:
                        temp_list.append([i0, all_pixels[1][i1[1] + 1][i1[0] + 1]])
                        break

        for i0 in range(len(seg_oline[1])):
            for i1 in seg_oline[1][i0]:
                if (i1[0] - 1 > -1) and all_pixels[0][i1[1]][i1[0] - 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1]][i1[0] - 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1]][i1[0] - 1], i0])
                        break
                if (i1[0] + 1 < len(all_pixels[0][i1[1]])) and all_pixels[0][i1[1]][i1[0] + 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1]][i1[0] + 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1]][i1[0] + 1], i0])
                        break
                if (i1[1] - 1 > -1) and all_pixels[0][i1[1] - 1][i1[0]] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] - 1][i1[0]], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] - 1][i1[0]], i0])
                        break
                if (i1[1] + 1 < len(all_pixels[0])) and all_pixels[0][i1[1] + 1][i1[0]] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] + 1][i1[0]], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] + 1][i1[0]], i0])
                        break
                if (i1[0] - 1 > -1) and (i1[1] - 1 > -1) and all_pixels[0][i1[1] - 1][i1[0] - 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] - 1][i1[0] - 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] - 1][i1[0] - 1], i0])
                        break
                if (i1[0] + 1 < len(all_pixels[0][i1[1] - 1])) and (i1[1] - 1 > -1) and all_pixels[0][i1[1] - 1][i1[0] + 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] - 1][i1[0] + 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] - 1][i1[0] + 1], i0])
                        break
                if (i1[0] - 1 > -1) and (i1[1] + 1 < len(all_pixels[0])) and all_pixels[0][i1[1] + 1][i1[0] - 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] + 1][i1[0] - 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] + 1][i1[0] - 1], i0])
                        break
                if (i1[1] + 1 < len(all_pixels[0])) and (i1[0] + 1 < len(all_pixels[0][i1[1] + 1])) and all_pixels[0][i1[1] + 1][i1[0] + 1] != -1:
                    if temp_list.__contains__([all_pixels[0][i1[1] + 1][i1[0] + 1], i0]) == False:
                        temp_list.append([all_pixels[0][i1[1] + 1][i1[0] + 1], i0])
                        break
    
    # generate image for an outlines
    oline_pixels = [[] for i in range(img_height)]
    for i in range(img_height):
        oline_pixels[i] = [np.float32([0, 0, 0]) for i in range(img_width)]

    """for j1 in range(len(seg_oline[0])):
        for j2 in range(len(seg_oline[0][j1])):
            oline_pixels[seg_oline[0][j1][j2][1]][seg_oline[0][j1][j2][0]] = np.float32([255, 255, 255])
    for j1 in range(len(seg_oline[1])):
        for j2 in range(len(seg_oline[1][j1])):
            oline_pixels[seg_oline[1][j1][j2][1]][seg_oline[1][j1][j2][0]] = np.float32([0, 255, 255])"""
    
    """for j1 in range(len(seg_vld[0])):
        for j2 in range(len(seg_vld[0][j1])):
            oline_pixels[seg_vld[0][j1][j2][1]][seg_vld[0][j1][j2][0]] = np.float32([255, 255, 255])
    for j1 in range(len(seg_vld[1])):
        for j2 in range(len(seg_vld[1][j1])):
            oline_pixels[seg_vld[1][j1][j2][1]][seg_vld[1][j1][j2][0]] = np.float32([0, 255, 255])"""

    #oline_pixels = np.array(oline_pixels)  # pred_pixels
    #oline_pixels = oline_pixels.astype(np.uint8)
    #plt.imshow(oline_pixels)
    #plt.show()
    
    """# Add pixels of Class 2 to Class 1
    for i in range(len(temp_list)):
        for j in seg_vld[1][temp_list[i][1] - i]:
            seg_vld[0][temp_list[i][0]].append(j)
            all_pixels[1][j[1]][j[0]] = -1
        del seg_oline[1][temp_list[i][1] - i] # - i depicts updated position of temp_list[i][1] after each deletion
        del seg_vld[1][temp_list[i][1] - i]"""

    print('len(seg_vld): ', len(seg_vld[0]), len(seg_vld[1]))
    seg_init_groups = []
    if len(temp_list) > 0:
        #print('temp_list[0]:', temp_list[0])

        # GROUP NEIGHBOURING SEGMENTS as per 'temp_list'
        seg_init_groups = [[[temp_list[0][0]], [temp_list[0][1]]]] # idx_class_1, idx_class_2
        for i1 in range(1, len(temp_list)):
            flag2 = False
            for i2 in range(len(seg_init_groups)):
                if seg_init_groups[i2][0].__contains__(temp_list[i1][0]) == True:
                    if seg_init_groups[i2][1].__contains__(temp_list[i1][1]) == False:
                        seg_init_groups[i2][1].append(temp_list[i1][1])
                        flag2 = True
                elif seg_init_groups[i2][1].__contains__(temp_list[i1][1]) == True:
                    if seg_init_groups[i2][0].__contains__(temp_list[i1][0]) == False:
                        seg_init_groups[i2][0].append(temp_list[i1][0])
                        flag2 = True
                #if temp_list[i1][0] == 54:
                #    print(f'found 54: {temp_list[i1]}  {seg_init_groups[i2]}  {flag2}')
            if flag2 == False:
                seg_init_groups.append([[temp_list[i1][0]], [temp_list[i1][1]]])
        
        j1 = 0
        while (j1 < len(seg_init_groups) - 1):
            j2 = j1 + 1
            while (j2 < len(seg_init_groups)):
                flag2 = False
                for i1 in range(len(seg_init_groups[j2][0])):
                    if (i1 < len(seg_init_groups[j2][0])) and seg_init_groups[j1][0].__contains__(seg_init_groups[j2][0][i1]) == True:
                        flag2 = True
                        for i2 in range(len(seg_init_groups[j2][0])):
                            if seg_init_groups[j1][0].__contains__(seg_init_groups[j2][0][i2]) == False:
                                seg_init_groups[j1][0].append(seg_init_groups[j2][0][i2])
                        for i2 in range(len(seg_init_groups[j2][1])):
                            if seg_init_groups[j1][1].__contains__(seg_init_groups[j2][1][i2]) == False:
                                seg_init_groups[j1][1].append(seg_init_groups[j2][1][i2])

                    elif (i1 < len(seg_init_groups[j2][1])) and seg_init_groups[j1][1].__contains__(seg_init_groups[j2][1][i1]) == True:
                        flag2 = True
                        for i2 in range(len(seg_init_groups[j2][0])):
                            if seg_init_groups[j1][0].__contains__(seg_init_groups[j2][0][i2]) == False:
                                seg_init_groups[j1][0].append(seg_init_groups[j2][0][i2])
                        for i2 in range(len(seg_init_groups[j2][1])):
                            if seg_init_groups[j1][1].__contains__(seg_init_groups[j2][1][i2]) == False:
                                seg_init_groups[j1][1].append(seg_init_groups[j2][1][i2])
                if flag2 == True:
                    del seg_init_groups[j2]
                else:
                    j2 += 1
            j1 += 1
        
        
        for j1 in range(len(seg_init_groups)):
            seg_init_groups[j1][0] = sorted(seg_init_groups[j1][0])
            seg_init_groups[j1][1] = sorted(seg_init_groups[j1][1])
    
    
    ## If a 'seg_init_groups[j1]' has only single segment number in [0] and [1],
    ## then find the segment with max pixels (also indicates individual tree / cluster) and store all pixels in that segment.
    ## Set 'seg_t_f_vld' value of the smaller segment as 'False'.
    
    ## For 'seg_init_groups[j1]' having multiple segments in [0] and [1],
    ## then similar condition applies for collected pixel count of individual trees and clusters for the cluster.
    
    seg_grp_cn = [] # pixel count of segment groups
    for line1 in seg_init_groups:
        temp_0, temp_1 = 0, 0
        for i1 in line1[0]:
            temp_0 += len(seg_vld[0][i1])
        for i1 in line1[1]:
            temp_1 += len(seg_vld[1][i1])
        seg_grp_cn.append([temp_0, temp_1])
        #print(f'{line1}  pixel count: [{temp_0}, {temp_1}]\n')
    
    # put pixels of smaller groups to the largest group
    for j1 in range(len(seg_grp_cn)):
        if seg_grp_cn[j1][0] >= seg_grp_cn[j1][1]:
            for i1 in range(1, len(seg_init_groups[j1][0])):
                for i2 in range(len(seg_vld[0][seg_init_groups[j1][0][i1]])):
                    seg_vld[0][seg_init_groups[j1][0][0]].append(seg_vld[0][seg_init_groups[j1][0][i1]][i2])
                seg_vld[0][seg_init_groups[j1][0][i1]] = [] # commented only for testing
            
            for i1 in range(len(seg_init_groups[j1][1])):
                for i2 in range(len(seg_vld[1][seg_init_groups[j1][1][i1]])):
                    seg_vld[0][seg_init_groups[j1][0][0]].append(seg_vld[1][seg_init_groups[j1][1][i1]][i2])
                seg_vld[1][seg_init_groups[j1][1][i1]] = []
            
        else:
            for i1 in range(1, len(seg_init_groups[j1][1])):
                for i2 in range(len(seg_vld[1][seg_init_groups[j1][1][i1]])):
                    seg_vld[1][seg_init_groups[j1][1][0]].append(seg_vld[1][seg_init_groups[j1][1][i1]][i2])
                seg_vld[1][seg_init_groups[j1][1][i1]] = []
            
            for i1 in range(len(seg_init_groups[j1][0])):
                for i2 in range(len(seg_vld[0][seg_init_groups[j1][0][i1]])):
                    seg_vld[1][seg_init_groups[j1][1][0]].append(seg_vld[0][seg_init_groups[j1][0][i1]][i2])
                seg_vld[0][seg_init_groups[j1][0][i1]] = []

    """for j1 in range(len(seg_vld[0])):
        print(f'{j1}: len:{len(seg_vld[0][j1])},', end= "  ")
    print()
    for j1 in range(len(seg_vld[1])):
        print(f'{j1}: len:{len(seg_vld[1][j1])},', end= "  ")
    print()"""

    max_seg_vld_0 = 0
    if len(seg_vld[0]) > 0:
        max_seg_vld_0 = max([len(seg_vld[0][i]) for i in range(len(seg_vld[0]))])
        min_seg_vld_0 = max_seg_vld_0
        for i1 in range(len(seg_vld[0])):
            if seg_vld[0][i1] != [] and len(seg_vld[0][i1]) < min_seg_vld_0:
                min_seg_vld_0 = len(seg_vld[0][i1])
    max_seg_vld_1 = 0
    if len(seg_vld[1]) > 0:
        max_seg_vld_1 = max([len(seg_vld[1][i]) for i in range(len(seg_vld[1]))])
        min_seg_vld_1 = max_seg_vld_1
        for i1 in range(len(seg_vld[1])):
            if seg_vld[1][i1] != [] and len(seg_vld[1][i1]) < min_seg_vld_1:
                min_seg_vld_1 = len(seg_vld[1][i1])
    
    if max_seg_vld_0 > 0:
        print('Individual tree area: min, max, min/max: ', min_seg_vld_0, max_seg_vld_0, min_seg_vld_0 / max_seg_vld_0)
    if max_seg_vld_1 > 0:
        print('Cluster area: min, max, min/max: ', min_seg_vld_1, max_seg_vld_1, min_seg_vld_1 / max_seg_vld_1)
    if max_seg_vld_0 > 0:
        min_accepted_a0 = float(input('\nEnter minimum accepted area for individual trees (in %): '))
    if max_seg_vld_1 > 0:
        min_accepted_a1 = float(input('\nEnter minimum accepted area for cluster of trees (in %): '))

    # generate image for the grouped segments
    pred_segment = [[] for i in range(img_height)]
    for i in range(img_height):
        pred_segment[i] = [np.float32([0, 0, 0]) for i in range(img_width)]

    #cmap   = plt.cm.get_cmap("tab10")
    cmap   = plt.get_cmap("tab10")
    for i in range(1, num_classes):
        r, g, b, _ = cmap((i - 1) % 10)
        if i == 1:
            for j1 in range(len(seg_vld[0])):
                if len(seg_vld[0][j1]) >= min_accepted_a0 * max_seg_vld_0:
                    for j2 in range(len(seg_vld[0][j1])):
                        pred_segment[seg_vld[0][j1][j2][1]][seg_vld[0][j1][j2][0]] = [int(r * 255), int(g * 255), int(b * 255)]
        else:
            for j1 in range(len(seg_vld[1])):
                if len(seg_vld[1][j1]) >= min_accepted_a1 * max_seg_vld_1:
                    for j2 in range(len(seg_vld[1][j1])):
                        pred_segment[seg_vld[1][j1][j2][1]][seg_vld[1][j1][j2][0]] = [int(r * 255), int(g * 255), int(b * 255)]

    pred_segment = np.array(pred_segment)  # pred_pixels
    pred_segment = pred_segment.astype(np.uint8)
    #plt.imshow(pred_segment)
    #plt.show()
    class_fname = out_dir / f"{stem}_mask_final.png"
    # cv2 expects BGR — convert from RGB
    cv2.imwrite(str(class_fname),
                cv2.cvtColor(pred_segment, cv2.COLOR_RGB2BGR))

    #  Terminal pixel statistics 
    total = pred_map.size
    print(f"\n  Pixel statistics for : {wi_name}")
    print(f"  {'Class':<20} {'Pixels':>10}   {'Area %':>7}")
    print(f"  {'-'*44}")
    for c, name in enumerate(class_names):
        count = int((pred_map == c).sum())
        print(f"  {name:<20} {count:>10,}   {count / total * 100:>6.2f}%")

    return


#  Optional Matplotlib Summary Figure

def save_summary_figure(wi:          np.ndarray,
                        pred_map:     np.ndarray,
                        wi_name:     str,
                        out_dir:      Path,
                        num_classes:  int,
                        class_colors: np.ndarray,
                        class_names:  list):
    """
    Saves a single matplotlib figure that shows:
      Row 0 : original wi  |  composite overlay
      Row 1 : one binary mask panel per foreground class
    """
    composite    = build_composite(wi, pred_map, class_colors)
    binary_masks = build_binary_masks(pred_map, num_classes)
    fg_count     = num_classes - 1

    ncols_r1 = max(2, fg_count)
    fig_w    = max(12, 5 * ncols_r1)
    fig, axes = plt.subplots(2, ncols_r1, figsize=(fig_w, 10))

    #  Row 0 
    axes[0][0].imshow(wi)
    axes[0][0].set_title(
        f"Original wi  ({wi.shape[1]}x{wi.shape[0]})",
        fontsize=11, fontweight="bold")
    axes[0][0].axis("off")

    axes[0][1].imshow(composite)
    axes[0][1].set_title("Composite overlay",
                          fontsize=11, fontweight="bold")
    axes[0][1].axis("off")
    axes[0][1].legend(
        handles=[
            mpatches.Patch(
                facecolor=tuple(v / 255 for v in class_colors[c][:3]),
                alpha=class_colors[c][3] / 255,
                label=class_names[c]
            )
            for c in range(1, num_classes)
        ],
        loc="lower right", fontsize=9, framealpha=0.85
    )
    for j in range(2, ncols_r1):
        axes[0][j].axis("off")

    #  Row 1 — binary masks 
    for i, (c, mask) in enumerate(
            zip(range(1, num_classes), binary_masks)):
        if i >= ncols_r1:
            break
        axes[1][i].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1][i].set_title(
            f"Class {c} binary mask",
            fontsize=11, fontweight="bold")
        axes[1][i].axis("off")

    for j in range(fg_count, ncols_r1):
        axes[1][j].axis("off")

    stem     = Path(wi_name).stem
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stem}_summary_default.png" # f"{stem}_summary_default_{ts}.png"

    plt.suptitle(f"ConvUNeXt wi Inference — {stem}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary figure  -> {out_path}")


#  Collect wi Inputs

def collect_inputs() -> list:
    p = Path(INPUT)
    if p.is_file():
        if p.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        return [p]
    elif p.is_dir():
        images = sorted([
            f for f in p.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ])
        if not images:
            raise FileNotFoundError(f"No supported images found in: {p}")
        return images
    else:
        raise FileNotFoundError(f"INPUT path does not exist: {p}")


#  Entry Point

def main():
    providers = select_providers()

    print("\n" + "=" * 60)
    print("  ConvUNeXt — Whole Slide Image Inference")
    print("=" * 60)

    session, img_size, num_classes, input_name, output_name = \
        load_session(providers)

    class_colors = make_class_colors(num_classes, alpha=OVERLAY_ALPHA)
    class_names  = make_class_names(num_classes)
    stride       = img_size // 3

    wi_paths = collect_inputs()
    print(f"\n  WIs to process : {len(wi_paths)}")
    print(f"  Classes         : {num_classes}  {class_names}")
    print(f"  Patch size      : {img_size} x {img_size}")
    print(f"  Stride          : {stride} px")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for wi_path in wi_paths:
        print(f"\n{'='*60}")
        print(f"  {wi_path.name}")
        print(f"{'='*60}")

        wi = load_image_rgb(str(wi_path))
        m1, m2 = wi.shape[:2]
        print(f"  wi dimensions : {m2} x {m1}  (W x H)")

        # Each wi gets its own output subfolder
        wi_out_dir = Path(OUTPUT_DIR) / wi_path.stem
        wi_out_dir.mkdir(parents=True, exist_ok=True)

        #  Sliding window inference 
        pred_map = infer_wi(
            session, wi, img_size, num_classes,
            input_name, output_name
        )

        #  Save n output files 
        save_outputs(
            wi          = wi,
            pred_map     = pred_map,
            wi_name     = wi_path.name,
            out_dir      = wi_out_dir,
            num_classes  = num_classes,
            class_colors = class_colors,
            class_names  = class_names,
        )

        #  Save matplotlib summary figure 
        save_summary_figure(
            wi          = wi,
            pred_map     = pred_map,
            wi_name     = wi_path.name,
            out_dir      = wi_out_dir,
            num_classes  = num_classes,
            class_colors = class_colors,
            class_names  = class_names,
        )

    print("\n" + "=" * 60)
    print(f"  Done.  Processed {len(wi_paths)} wi(s).")
    print(f"  Results saved to : {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
