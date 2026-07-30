import json
import numpy as np
import cv2
import math
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
seg_vld = [
    [[[4,5],[5,6]], [[12,13],[12,14],[13,14]]],
    [[[44,5],[45,6]], [[62,13],[62,14],[63,14]]]
]
"""
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

def short_dist(point1, point2): # shortest distance
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    
    squared_diff_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    return math.sqrt(squared_diff_sum)

#  Entry Point
#def main():
def main(out_path, INPUT, OVERLAY_ALPHA):
    #wi_path = r"convunext\test_wi\4.png"
    wi_path = INPUT
    wi = load_image_rgb(str(wi_path))
    img_height, img_width = wi.shape[:2]

    #json_path = r'F:\Traditional Machine learning project tree forest\convunext\test_output_whole\4\4_seg_raw.json'
    json_path = str(out_path) + '/' + str(Path(out_path).stem) + '_seg_raw.json'
    with open(json_path, 'r') as f:
        seg_vld = json.load(f)

    valid_rows, valid_columns = [], []
    #json_path = r'F:\Traditional Machine learning project tree forest\convunext\test_output_whole\4\4_valid_r_c.json'
    json_path = str(out_path) + '/' + str(Path(out_path).stem) + '_valid_r_c.json'
    with open(json_path, 'r') as f:
        valid_r_c = json.load(f)
    valid_rows = [valid_r_c[0][0], valid_r_c[1][0]]
    valid_columns = [valid_r_c[0][1], valid_r_c[1][1]]
    #print(valid_r_c[0], valid_r_c[1])
    print(f'valid_rows: {valid_rows}; valid_columns: {valid_columns}')


    # Use it exactly like before
    #print(seg_vld[0]) # First image/frame: [[[4,5],[5,6]], [[12,13]...]]
    ## print(seg_vld[0][1])
    #print(seg_vld[0][1][2]) # -> [13, 14]
    #print(seg_vld[1][0][1]) # -> [45, 6]

    # FIND SEGMENT OUTLINE
    seg_oline, all_pixels = [], []
    for k in range(len(seg_vld)):
        all_pixels.append([])
        all_pixels[-1] = [[] for i in range(img_height)]
        for i in range(img_height):
            all_pixels[-1][i] = [-1 for j in range(img_width)]

        for j1 in range(len(seg_vld[k])):
            for j2 in range(len(seg_vld[k][j1])):
                all_pixels[-1][seg_vld[k][j1][j2][1]][seg_vld[k][j1][j2][0]] = j1

        # FIND SEGMENT OUTLINE
        seg_oline.append([])
        seg_oline[-1] = [[] for i in range(len(seg_vld[k]))]

        if valid_columns[k][1] >= valid_columns[k][0]:
            # First row
            if all_pixels[-1][valid_rows[k][0]][valid_columns[k][0]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[k][0]][valid_columns[k][0]]].append([valid_columns[k][0], valid_rows[k][0]])
                
            for i2 in range(valid_columns[k][0] + 1, valid_columns[k][1]): # valid_rows[0]
                if all_pixels[-1][valid_rows[k][0]][i2] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][valid_rows[k][0]][i2]].append([i2, valid_rows[k][0]])

            if all_pixels[-1][valid_rows[k][0]][valid_columns[k][1]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[k][0]][valid_columns[k][1]]].append([valid_columns[k][1], valid_rows[k][0]])
            
            # Rows in between first and last
            for i1 in range(valid_rows[k][0] + 1, valid_rows[k][1]):
                if all_pixels[-1][i1][valid_columns[k][0]] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][i1][valid_columns[k][0]]].append([valid_columns[k][0], i1])

                for i2 in range(valid_columns[k][0] + 1, valid_columns[k][1]):
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

                if all_pixels[-1][i1][valid_columns[k][1]] > -1: # valid pixel (not background)
                    #print(all_pixels[-1][i1][valid_columns[k][1]], len(seg_oline[-1]) - 1)
                    seg_oline[-1][all_pixels[-1][i1][valid_columns[k][1]]].append([valid_columns[k][1], i1])
            
            # Last row
            if all_pixels[-1][valid_rows[k][1]][valid_columns[k][0]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[k][1]][valid_columns[k][0]]].append([valid_columns[k][0], valid_rows[k][1]])
                
            for i2 in range(valid_columns[k][0] + 1, valid_columns[k][1]): # valid_rows[0]
                if all_pixels[-1][valid_rows[k][1]][i2] > -1: # valid pixel (not background)
                    seg_oline[-1][all_pixels[-1][valid_rows[k][1]][i2]].append([i2, valid_rows[k][1]])

            if all_pixels[-1][valid_rows[k][1]][valid_columns[k][1]] > -1: # valid pixel (not background)
                seg_oline[-1][all_pixels[-1][valid_rows[k][1]][valid_columns[k][1]]].append([valid_columns[k][1], valid_rows[k][1]])


    # FIND MIN RADIUS AND CENTER OF SEGMENTS
    seg_center, seg_radius = [], []
    for k in range(len(seg_oline)):
        seg_center.append([])
        seg_radius.append([])
        for j1 in range(len(seg_oline[k])):
            seg_center[-1].append([])
            seg_radius[-1].append(0)
            x1, y1 = 0, 0
            if len(seg_oline[k][j1]) > 0:
                for j2 in range(len(seg_oline[k][j1])):
                    x1 += seg_oline[k][j1][j2][0]
                    y1 += seg_oline[k][j1][j2][1]
                x1 = round(x1 / len(seg_oline[k][j1]))
                y1 = round(y1 / len(seg_oline[k][j1]))
    
                seg_center[-1][-1] = [x1, y1]
                temp_radius = []
                for j2 in range(len(seg_oline[k][j1])):
                    temp_radius.append(short_dist([seg_oline[k][j1][j2][0], seg_oline[k][j1][j2][1]],
                                                seg_center[-1][-1]))
                seg_radius[-1][-1] = min(temp_radius) #statistics.mean(temp_radius)

    #print('center: ', seg_center)
    #print('\nradius: ', seg_radius)


    # MAP the seg indices in a grid
    seg_map0, seg_map1 = [], []
    for i1 in range(img_width):
        seg_map0.append([])
        seg_map1.append([])
        for i2 in range(img_height):
            seg_map0[-1].append(-1)
            seg_map1[-1].append(-1)
    
    for i1 in range(len(seg_oline[0])):
        if seg_radius[0][i1] != 0:
            for line in seg_oline[0][i1]:
                seg_map0[line[0]][line[1]] = i1
    
    for i1 in range(len(seg_oline[1])):
        if seg_radius[1][i1] != 0:
            for line in seg_oline[1][i1]:
                seg_map1[line[0]][line[1]] = i1
    

    # FIND NEIGHBOURS from seg_outline  (consider segments less than 10% of maximum)
    min_area_perc = 0.01
    seg0_max, seg1_max = 0, 0
    for i1 in range(len(seg_vld[0])):
        if seg_radius[0][i1] != 0:
            if len(seg_vld[0][i1]) > seg0_max:
                seg0_max = len(seg_vld[0][i1])
    for i1 in range(len(seg_vld[1])):
        if seg_radius[1][i1] != 0:
            if len(seg_vld[1][i1]) > seg1_max:
                seg1_max = len(seg_vld[1][i1])

    seg0_min, seg1_min = seg0_max, seg1_max
    for i1 in range(len(seg_vld[0])):
        if seg_radius[0][i1] != 0:
            if len(seg_vld[0][i1]) < seg0_min:
                seg0_min = len(seg_vld[0][i1])
    for i1 in range(len(seg_vld[1])):
        if seg_radius[1][i1] != 0:
            if len(seg_vld[1][i1]) < seg1_min:
                seg1_min = len(seg_vld[1][i1])
    print('\nseg0_max, seg0_min, seg1_max, seg1_min: ', seg0_max, seg0_min, seg1_max, seg1_min)

    final_min_vld_1_area = float(input('\nEnter minimum cluster area (in %) (E.g. 10.5): '))
    final_min_vld_1_area = final_min_vld_1_area / 100

    correct_seg0, correct_seg1, correct_seg1_indv = [], [], [] # store the list of indices

    for i1 in range(len(seg_oline[0])):
        correct_seg0.append(i1)
    #print(f'\n{correct_seg0}')

    for i1 in range(len(seg_oline[1])):
        if len(seg_vld[1][i1]) < (final_min_vld_1_area * seg1_max):
            correct_seg1_indv.append(i1)
        else:
            correct_seg1.append(i1)
    #print(f'\n{correct_seg1} {correct_seg1_indv}')

    # save image
    # generate image for the grouped segments
    pred_segment = [[] for i in range(img_height)]
    for i in range(img_height):
        pred_segment[i] = [np.float32([0, 0, 0]) for i in range(img_width)]

    tree_pixel_cn, cluster_pixel_cn = 0, 0 
    cmap   = plt.get_cmap("tab10")
    for i in range(1, len(seg_vld) + 1): # + 1 since the summation should result in 3
        r1, g1, b1, _ = cmap((i - 1) % 10)
        r1, g1, b1 = int(r1 * 255), int(g1 * 255), int(b1 * 255)
        if i == 1:
            for j1 in correct_seg0:
                for j2 in range(len(seg_vld[0][j1])):
                    pred_segment[seg_vld[0][j1][j2][1]][seg_vld[0][j1][j2][0]] = [r1, g1, b1]
                    tree_pixel_cn += 1
            for j1 in correct_seg1_indv:
                for j2 in range(len(seg_vld[1][j1])):
                    pred_segment[seg_vld[1][j1][j2][1]][seg_vld[1][j1][j2][0]] = [r1, g1, b1]
                    tree_pixel_cn += 1
        else:
            r2, g2, b2, _ = cmap((i - 1) % 10)
            r2, g2, b2 = int(r2 * 255), int(g2 * 255), int(b2 * 255)
            for j1 in correct_seg1:
                for j2 in range(len(seg_vld[1][j1])):
                    pred_segment[seg_vld[1][j1][j2][1]][seg_vld[1][j1][j2][0]] = [r2, g2, b2]
                    cluster_pixel_cn += 1

    pred_segment = np.array(pred_segment)  # pred_pixels
    stem = Path(out_path).stem
    with open(str(out_path / f"{stem}_seg_final.json"), 'w') as f:
        json.dump(seg_vld, f, indent=2)
    pred_segment = pred_segment.astype(np.uint8)
    
    # --- Load background (BGR, uint8, no alpha channel) ---
    background = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)  # shape: (img_height, img_width, 3)

    # --- pred_segment is currently RGB (from matplotlib's cmap) ---
    # cv2 works in BGR, so convert before blending
    pred_segment_bgr = cv2.cvtColor(pred_segment, cv2.COLOR_RGB2BGR)

    # --- Mask: True wherever a segment color was assigned (i.e. not [0,0,0]) ---
    mask = np.any(pred_segment != 0, axis=-1)  # shape: (img_height, img_width), bool

    # --- Alpha blend the whole image ---
    alpha = OVERLAY_ALPHA / 255.0  # normalize 0-255 -> 0-1
    blended = cv2.addWeighted(pred_segment_bgr, alpha, background, 1 - alpha, 0)

    # --- Combine: use blended pixels where mask is True, original background elsewhere ---
    result = np.where(mask[:, :, None], blended, background).astype(np.uint8)

    #plt.imshow(pred_segment)
    #plt.show()
    #out_dir = Path(r"F:\Traditional Machine learning project tree forest\convunext\test_output_whole\4")
    #stem = '4'
    class_fname =  out_path / f"{stem}_mask_final.png"
    # cv2 expects BGR — convert from RGB
    cv2.imwrite(str(class_fname), cv2.cvtColor(pred_segment, cv2.COLOR_RGB2BGR))
    
    class_fname =  out_path / f"{stem}_mask_final_overlay.png"
    print(class_fname)
    cv2.imwrite(str(class_fname), result)

    #  Terminal pixel statistics
    print(f"\n  Final pixel statistics for : {Path(INPUT).name}")
    print(f"  {'Class':<20} {'Pixels':>10}   {'Area %':>7}")
    print(f"  {'-'*44}")
    print(f"  {str('Background'):<20} {((img_height * img_width) - (tree_pixel_cn + cluster_pixel_cn)):>10,}   {((img_height * img_width) - (tree_pixel_cn + cluster_pixel_cn)) / (img_height * img_width) * 100:>6.2f}%")
    print(f"  {str('Tree'):<20} {tree_pixel_cn:>10,}   {tree_pixel_cn / (img_height * img_width) * 100:>6.2f}%")
    print(f"  {str('Cluster'):<20} {cluster_pixel_cn:>10,}   {cluster_pixel_cn / (img_height * img_width) * 100:>6.2f}%")

    


#if __name__ == "__main__":
#    main()