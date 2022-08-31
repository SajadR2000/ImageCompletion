import cv2
import numpy as np
import matplotlib.pyplot as plt


# The code for this question is very similar to the previous question so I skipped commenting most of it.
def minimum_cut_path(overlapping_1, overlapping_2, cut_shape, max_allowable_movement):
    if cut_shape == 'horizontal_cut':
        # For horizontal minimum cut path, first transpose the matrices, then do a vertical minimum cut path
        overlapping_1 = overlapping_1.transpose((1, 0, 2))
        overlapping_2 = overlapping_2.transpose((1, 0, 2))
    overlapping_1 = overlapping_1.reshape((overlapping_1.shape[0], overlapping_1.shape[1], -1))
    overlapping_2 = overlapping_2.reshape((overlapping_2.shape[0], overlapping_2.shape[1], -1))
    cost = np.square(overlapping_1 - overlapping_2).sum(axis=2)

    # Vertical Minimum Cut Path
    path_cost = {"Path": [[i] for i in range(overlapping_1.shape[1])],
                 "Cost": [cost[0, i] for i in range(overlapping_1.shape[1])]}

    for i in range(1, overlapping_1.shape[0]):
        path_cost_temp = {"Path": [[] for i in range(overlapping_1.shape[1])],  # Resetting path_cost_temp
                          "Cost": [0 for i in range(overlapping_1.shape[1])]}
        for j in range(overlapping_1.shape[1]):
            if j < max_allowable_movement:
                neighbors_cost = path_cost["Cost"][:j + max_allowable_movement + 1]
                ii = np.argmin(neighbors_cost)
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
            elif max_allowable_movement <= j < overlapping_1.shape[1] - max_allowable_movement:
                neighbors_cost = path_cost["Cost"][j - max_allowable_movement:j + max_allowable_movement + 1]
                ii = np.argmin(neighbors_cost) + j - max_allowable_movement
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
            else:
                neighbors_cost = path_cost["Cost"][j - max_allowable_movement:]
                ii = np.argmin(neighbors_cost) + j - max_allowable_movement
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
        path_cost = path_cost_temp  # Copy doesn't work for complex data types. you can use copy.deepcopy(). I reset it.

    min_cost_idx = np.argmin(path_cost["Cost"])
    return path_cost["Path"][min_cost_idx]


def find_matching_patch_O_shape(texture,
                                overlap_l,
                                overlap_u,
                                overlap_r,
                                overlap_d,
                                patch_shape,
                                patch_top_left,
                                hole_top_left,  # Can't choose from holes, because their zero
                                hole_shape,
                                hole_top_left2=None,
                                hole_shape2=None,
                                hole_top_left3=None,
                                hole_shape3=None,
                                ):
    number_of_best_matches = 10
    ssd_l = cv2.matchTemplate(texture, overlap_l, cv2.TM_SQDIFF_NORMED)  # Aligning left patch
    ssd_u = cv2.matchTemplate(texture, overlap_u, cv2.TM_SQDIFF_NORMED)  # Aligning up patch
    ssd_r = cv2.matchTemplate(texture, overlap_r, cv2.TM_SQDIFF_NORMED)  # Aligning right patch
    ssd_d = cv2.matchTemplate(texture, overlap_d, cv2.TM_SQDIFF_NORMED)  # Aligning down patch
    nonoverlap_size = patch_shape[1] - overlap_r.shape[1]
    ssd_r = ssd_r[:, nonoverlap_size:]  # Start of the right and down overlapping part is different from left and up
    nonoverlap_size = patch_shape[0] - overlap_d.shape[0]  # So they where shifted
    ssd_d = ssd_d[nonoverlap_size:, :]
    t_ = np.max((ssd_d.max(), ssd_r.max(), ssd_l.max(), ssd_u.max())) + 100
    ssd_l[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_u[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_r[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_d[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_

    if hole_top_left2 is not None:
        st_ = hole_top_left2[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_u[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_r[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_d[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
    if hole_top_left3 is not None:
        st_ = hole_top_left3[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_u[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_r[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_d[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_

    w = np.min((ssd_u.shape[1], ssd_l.shape[1], ssd_r.shape[1], ssd_d.shape[1]))
    h = np.min((ssd_u.shape[0], ssd_l.shape[0], ssd_r.shape[0], ssd_d.shape[1]))

    ssd_l = ssd_l[:h, :w]
    ssd_u = ssd_u[:h, :w]
    ssd_r = ssd_r[:h, :w]
    ssd_d = ssd_d[:h, :w]

    ssd = ssd_r + ssd_l + ssd_u + ssd_d
    ssd[patch_top_left[0]-patch_shape[0]:patch_top_left[0]+patch_shape[1],
        patch_top_left[1]-patch_shape[1]:patch_top_left[1]+patch_shape[1]] = t_
    ssd_flatten = ssd.ravel()
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    # chosen_points = ssd_argsort[-number_of_best_matches:]
    chosen_points = ssd_argsort[:number_of_best_matches]
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]].copy()
    return out


def find_matching_patch_U_shape(texture,
                                overlap_l,
                                overlap_u,
                                overlap_r,
                                patch_shape,
                                patch_top_left,
                                hole_top_left,
                                hole_shape,
                                hole_top_left2=None,
                                hole_shape2=None,
                                hole_top_left3=None,
                                hole_shape3=None,
                                ):
    number_of_best_matches = 10
    ssd_l = cv2.matchTemplate(texture, overlap_l, cv2.TM_SQDIFF_NORMED)
    ssd_u = cv2.matchTemplate(texture, overlap_u, cv2.TM_SQDIFF_NORMED)
    ssd_r = cv2.matchTemplate(texture, overlap_r, cv2.TM_SQDIFF_NORMED)
    nonoverlap_size = patch_shape[1] - overlap_r.shape[1]
    ssd_r = ssd_r[:, nonoverlap_size:]
    t_ = np.max((ssd_r.max(), ssd_l.max(), ssd_u.max())) + 100
    ssd_l[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_u[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_r[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    if hole_top_left2 is not None:
        st_ = hole_top_left2[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_u[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_r[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
    if hole_top_left3 is not None:
        st_ = hole_top_left3[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_u[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_r[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_

    w = np.min((ssd_u.shape[1], ssd_l.shape[1], ssd_r.shape[1]))
    h = np.min((ssd_u.shape[0], ssd_l.shape[0], ssd_r.shape[0]))

    ssd_l = ssd_l[:h, :w]
    ssd_u = ssd_u[:h, :w]
    ssd_r = ssd_r[:h, :w]
    ssd = ssd_r + ssd_l + ssd_u
    ssd[patch_top_left[0]-20:patch_top_left[0] + patch_shape[0]+20, patch_top_left[1]-20:patch_top_left[1] + patch_shape[1]+20] = t_
    ssd_flatten = ssd.ravel()
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    # chosen_points = ssd_argsort[-number_of_best_matches:]
    chosen_points = ssd_argsort[:number_of_best_matches]
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]].copy()
    return out


def find_matching_patch_C_shape(texture,
                                overlap_l,
                                overlap_u,
                                overlap_d,
                                patch_shape,
                                patch_top_left,
                                hole_top_left,
                                hole_shape,
                                hole_top_left2=None,
                                hole_shape2=None,
                                hole_top_left3=None,
                                hole_shape3=None,
                                ):
    number_of_best_matches = 10
    ssd_l = cv2.matchTemplate(texture, overlap_l, cv2.TM_SQDIFF_NORMED)
    ssd_u = cv2.matchTemplate(texture, overlap_u, cv2.TM_SQDIFF_NORMED)
    ssd_d = cv2.matchTemplate(texture, overlap_d, cv2.TM_SQDIFF_NORMED)
    nonoverlap_size = patch_shape[0] - overlap_d.shape[0]
    ssd_d = ssd_d[nonoverlap_size:, :]
    t_ = np.max((ssd_d.max(), ssd_l.max(), ssd_u.max())) + 100
    ssd_l[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_u[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    ssd_d[hole_top_left[0]-patch_shape[0]:hole_top_left[0]+hole_shape[0],
          hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_

    if hole_top_left2 is not None:
        st_ = hole_top_left2[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_u[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
        ssd_d[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
    if hole_top_left3 is not None:
        st_ = hole_top_left3[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd_l[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_u[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_
        ssd_d[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_

    w = np.min((ssd_u.shape[1], ssd_l.shape[1], ssd_d.shape[1]))
    h = np.min((ssd_u.shape[0], ssd_l.shape[0], ssd_d.shape[0]))

    ssd_l = ssd_l[:h, :w]
    ssd_u = ssd_u[:h, :w]
    ssd_d = ssd_d[:h, :w]
    ssd = ssd_d + ssd_l + ssd_u
    st_ = patch_top_left[0]-20
    if st_ < 0:
        st_ = 0
    ssd[st_:patch_top_left[0] + patch_shape[0]+20, patch_top_left[1]-20:patch_top_left[1] + patch_shape[1]+20] = t_
    ssd_flatten = ssd.ravel()
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    # chosen_points = ssd_argsort[-number_of_best_matches:]
    chosen_points = ssd_argsort[:number_of_best_matches]
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]].copy()
    return out


def find_matching_patch_L_shape(texture,
                                patch_overlapping_horizontal,
                                patch_overlapping_vertical,
                                patch_shape,
                                hole_top_left,
                                hole_shape,
                                hole_top_left2=None,
                                hole_shape2=None,
                                hole_top_left3=None,
                                hole_shape3=None,
                                ):
    number_of_best_matches = 10

    ssd_vertical = cv2.matchTemplate(texture, patch_overlapping_vertical, cv2.TM_SQDIFF_NORMED)
    ssd_horizontal = cv2.matchTemplate(texture, patch_overlapping_horizontal, cv2.TM_SQDIFF_NORMED)
    ssd_horizontal = ssd_horizontal[:-(patch_shape[0] - patch_overlapping_horizontal.shape[0]), :]
    ssd_vertical = ssd_vertical[:, :-(patch_shape[1] - patch_overlapping_vertical.shape[1])]
    t_ = np.max((ssd_vertical.max(), ssd_horizontal.max())) + 100
    ssd = ssd_vertical + ssd_horizontal
    st_ = hole_top_left[0]-patch_shape[0]
    if st_ < 0:
        st_ = 0
    ssd[st_:hole_top_left[0]+hole_shape[0],
        hole_top_left[1]-patch_shape[1]:hole_top_left[1]+hole_shape[1]] = t_
    if hole_top_left2 is not None:
        st_ = hole_top_left2[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd[st_:hole_top_left2[0] + hole_shape2[0],
              hole_top_left2[1] - patch_shape[1]:hole_top_left2[1] + hole_shape2[1]] = t_
    if hole_top_left3 is not None:
        st_ = hole_top_left3[0] - patch_shape[0]
        if st_ < 0:
            st_ = 0
        ssd[st_:hole_top_left3[0] + hole_shape3[0],
              hole_top_left3[1] - patch_shape[1]:hole_top_left3[1] + hole_shape3[1]] = t_

    ssd_flatten = ssd.ravel()
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    # chosen_points = ssd_argsort[-number_of_best_matches:]
    chosen_points = ssd_argsort[:number_of_best_matches]
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]].copy()

    return out


def merging(left_overlap,
            up_overlap,
            next_patch,
            cut_shape,
            max_allowable_movement,
            right_or_down_overlap=None,
            down_overlap=None):
    if cut_shape == 'L_shape_cut':
        horizontal_overlapping_up = up_overlap
        horizontal_overlapping_down = next_patch[:up_overlap.shape[0], :]

        vertical_overlapping_left = left_overlap
        vertical_overlapping_right = next_patch[:, :left_overlap.shape[1]]

        horizontal_cut = minimum_cut_path(horizontal_overlapping_up,
                                          horizontal_overlapping_down,
                                          'horizontal_cut',
                                          max_allowable_movement)

        vertical_cut = minimum_cut_path(vertical_overlapping_left,
                                        vertical_overlapping_right,
                                        'vertical_cut',
                                        max_allowable_movement)

        horizontal_overlapped = np.zeros(horizontal_overlapping_up.shape, dtype=np.uint8)
        for i in range(horizontal_overlapped.shape[1]):
            horizontal_overlapped[:horizontal_cut[i], i] = horizontal_overlapping_up[:horizontal_cut[i], i]
            horizontal_overlapped[horizontal_cut[i]:, i] = horizontal_overlapping_down[horizontal_cut[i]:, i]

        for i in range(horizontal_overlapped.shape[0]):
            horizontal_overlapped[i, :vertical_cut[i]] = horizontal_overlapping_up[i, :vertical_cut[i]]

        vertical_overlapped = np.zeros(vertical_overlapping_left.shape, dtype=np.uint8)
        for i in range(vertical_overlapped.shape[0]):
            vertical_overlapped[i, :vertical_cut[i]] = vertical_overlapping_left[i, :vertical_cut[i]]
            vertical_overlapped[i, vertical_cut[i]:] = vertical_overlapping_right[i, vertical_cut[i]:]

        for i in range(vertical_overlapped.shape[1]):
            vertical_overlapped[:horizontal_cut[i], i] = vertical_overlapping_left[:horizontal_cut[i], i]

        non_overlapping_next = next_patch[horizontal_overlapped.shape[0]:, vertical_overlapped.shape[1]:]
        next_patch_modified = np.concatenate(
            (horizontal_overlapped[:, vertical_overlapped.shape[1]:], non_overlapping_next), axis=0)
        next_patch_modified = np.concatenate((vertical_overlapped, next_patch_modified), axis=1)
        return next_patch_modified

    elif cut_shape == 'U_shape_cut':
        left_overlap_next = next_patch[:, :left_overlap.shape[1]]
        cut_left = minimum_cut_path(left_overlap, left_overlap_next, 'vertical_cut', max_allowable_movement)
        left_overlapped = np.zeros(left_overlap.shape, dtype=np.uint8)
        for i in range(left_overlapped.shape[0]):
            left_overlapped[i, :cut_left[i]] = left_overlap[i, :cut_left[i]]
            left_overlapped[i, cut_left[i]:] = left_overlap_next[i, cut_left[i]:]

        up_overlap_next = next_patch[:up_overlap.shape[0], :]
        cut_up = minimum_cut_path(up_overlap, up_overlap_next, 'horizontal_cut', max_allowable_movement)
        up_overlapped = np.zeros(up_overlap.shape, dtype=np.uint8)
        for i in range(up_overlapped.shape[1]):
            up_overlapped[:cut_up[i], i] = up_overlap[:cut_up[i], i]
            up_overlapped[cut_up[i]:, i] = up_overlap_next[cut_up[i]:, i]

        for i in range(left_overlapped.shape[1]):
            left_overlapped[:cut_up[i], i] = left_overlap[:cut_up[i], i]
        for i in range(up_overlapped.shape[0]):
            up_overlapped[i, :cut_left[i]] = up_overlap[i, :cut_left[i]]

        right_overlap = right_or_down_overlap
        right_overlap_next= next_patch[:, -right_overlap.shape[1]:]
        cut_right = minimum_cut_path(right_overlap_next, right_overlap, 'vertical_cut', max_allowable_movement)
        right_overlapped = np.zeros(right_overlap.shape, dtype=np.uint8)
        for i in range(right_overlapped.shape[0]):
            right_overlapped[i, :cut_right[i]] = right_overlap_next[i, :cut_right[i]]
            right_overlapped[i, cut_right[i]:] = right_overlap[i, cut_right[i]:]

        cut_up = cut_up[-right_overlapped.shape[1]:]
        for i in range(right_overlapped.shape[1]):
            right_overlapped[:cut_up[i], i] = right_overlap[:cut_up[i], i]
        for i in range(up_overlapped.shape[0]):
            up_overlapped[i, -(right_overlapped.shape[1] - cut_right[i]):] = \
                up_overlap[i, -(right_overlapped.shape[1] - cut_right[i])]

        next_patch_modified = next_patch
        next_patch_modified[:up_overlapped.shape[0], :] = up_overlapped
        next_patch_modified[:, :left_overlapped.shape[1]] = left_overlapped
        next_patch_modified[:, -right_overlapped.shape[1]:] = right_overlapped

        return next_patch_modified

    elif cut_shape == 'C_shape_cut':
        left_overlap_next = next_patch[:, :left_overlap.shape[1]]
        cut_left = minimum_cut_path(left_overlap, left_overlap_next, 'vertical_cut', max_allowable_movement)
        left_overlapped = np.zeros(left_overlap.shape, dtype=np.uint8)
        for i in range(left_overlapped.shape[0]):
            left_overlapped[i, :cut_left[i]] = left_overlap[i, :cut_left[i]]
            left_overlapped[i, cut_left[i]:] = left_overlap_next[i, cut_left[i]:]

        up_overlap_next = next_patch[:up_overlap.shape[0], :]
        cut_up = minimum_cut_path(up_overlap, up_overlap_next, 'horizontal_cut', max_allowable_movement)
        up_overlapped = np.zeros(up_overlap.shape, dtype=np.uint8)
        for i in range(up_overlapped.shape[1]):
            up_overlapped[:cut_up[i], i] = up_overlap[:cut_up[i], i]
            up_overlapped[cut_up[i]:, i] = up_overlap_next[cut_up[i]:, i]

        for i in range(left_overlapped.shape[1]):
            left_overlapped[:cut_up[i], i] = left_overlap[:cut_up[i], i]
        for i in range(up_overlapped.shape[0]):
            up_overlapped[i, :cut_left[i]] = up_overlap[i, :cut_left[i]]

        down_overlap = right_or_down_overlap.copy()
        down_overlap_next = next_patch[-down_overlap.shape[0]:, :]
        cut_down = minimum_cut_path(down_overlap_next, down_overlap, 'horizontal_cut', max_allowable_movement)
        down_overlapped = np.zeros(down_overlap.shape, dtype=np.uint8)
        for i in range(down_overlapped.shape[1]):
            down_overlapped[:cut_down[i], i] = down_overlap_next[:cut_down[i], i]
            down_overlapped[cut_down[i]:, i] = down_overlap[cut_down[i]:, i]

        cut_left = cut_left[-down_overlapped.shape[0]:]
        for i in range(down_overlapped.shape[0]):
            down_overlapped[i, :cut_left[i]] = down_overlap[i, :cut_left[i]]
        for i in range(left_overlapped.shape[1]):
            left_overlapped[-(down_overlapped.shape[0] - cut_down[i]):, i] = \
                left_overlap[-(down_overlapped.shape[0] - cut_down[i]):, i]

        next_patch_modified = next_patch.copy()
        next_patch_modified[:up_overlapped.shape[0], :] = up_overlapped
        next_patch_modified[:, :left_overlapped.shape[1]] = left_overlapped
        next_patch_modified[-down_overlapped.shape[0]:, :] = down_overlapped

        return next_patch_modified

    elif cut_shape == 'O_shape_cut':
        left_overlap_next = next_patch[:, :left_overlap.shape[1]]
        cut_left = minimum_cut_path(left_overlap, left_overlap_next, 'vertical_cut', max_allowable_movement)
        left_overlapped = np.zeros(left_overlap.shape, dtype=np.uint8)
        for i in range(left_overlapped.shape[0]):
            left_overlapped[i, :cut_left[i]] = left_overlap[i, :cut_left[i]]
            left_overlapped[i, cut_left[i]:] = left_overlap_next[i, cut_left[i]:]

        up_overlap_next = next_patch[:up_overlap.shape[0], :]
        cut_up = minimum_cut_path(up_overlap, up_overlap_next, 'horizontal_cut', max_allowable_movement)
        up_overlapped = np.zeros(up_overlap.shape, dtype=np.uint8)
        for i in range(up_overlapped.shape[1]):
            up_overlapped[:cut_up[i], i] = up_overlap[:cut_up[i], i]
            up_overlapped[cut_up[i]:, i] = up_overlap_next[cut_up[i]:, i]

        for i in range(left_overlapped.shape[1]):
            left_overlapped[:cut_up[i], i] = left_overlap[:cut_up[i], i]
        for i in range(up_overlapped.shape[0]):
            up_overlapped[i, :cut_left[i]] = up_overlap[i, :cut_left[i]]

        right_overlap = right_or_down_overlap
        right_overlap_next = next_patch[:, -right_overlap.shape[1]:]
        cut_right = minimum_cut_path(right_overlap_next, right_overlap, 'vertical_cut', max_allowable_movement)
        right_overlapped = np.zeros(right_overlap.shape, dtype=np.uint8)
        for i in range(right_overlapped.shape[0]):
            right_overlapped[i, :cut_right[i]] = right_overlap_next[i, :cut_right[i]]
            right_overlapped[i, cut_right[i]:] = right_overlap[i, cut_right[i]:]

        cut_up = cut_up[-right_overlapped.shape[1]:]
        for i in range(right_overlapped.shape[1]):
            right_overlapped[:cut_up[i], i] = right_overlap[:cut_up[i], i]
        for i in range(up_overlapped.shape[0]):
            up_overlapped[i, -(right_overlapped.shape[1] - cut_right[i]):] = \
                up_overlap[i, -(right_overlapped.shape[1] - cut_right[i])]

        down_overlap = down_overlap.copy()
        down_overlap_next = next_patch[-down_overlap.shape[0]:, :]
        cut_down = minimum_cut_path(down_overlap_next, down_overlap, 'horizontal_cut', max_allowable_movement)
        down_overlapped = np.zeros(down_overlap.shape, dtype=np.uint8)
        for i in range(down_overlapped.shape[1]):
            down_overlapped[:cut_down[i], i] = down_overlap_next[:cut_down[i], i]
            down_overlapped[cut_down[i]:, i] = down_overlap[cut_down[i]:, i]

        cut_left = cut_left[-down_overlapped.shape[0]:]
        for i in range(down_overlapped.shape[0]):
            down_overlapped[i, :cut_left[i]] = down_overlap[i, :cut_left[i]]
        for i in range(left_overlapped.shape[1]):
            left_overlapped[-(down_overlapped.shape[0] - cut_down[i]):, i] = \
                left_overlap[-(down_overlapped.shape[0] - cut_down[i]):, i]

        cut_right = cut_right[-down_overlapped.shape[0]:]
        for i in range(down_overlapped.shape[0]):
            down_overlapped[i, -(right_overlapped.shape[1] - cut_right[i]):] = \
                down_overlap[i, -(right_overlapped.shape[1] - cut_right[i]):]

        cut_down = cut_down[-right_overlapped.shape[1]:]
        for i in range(right_overlapped.shape[1]):
            right_overlapped[-(down_overlapped.shape[0] - cut_down[i]):, i] = \
                right_overlap[-(down_overlapped.shape[0] - cut_down[i]):, i]

        next_patch_modified = next_patch.copy()
        next_patch_modified[:up_overlapped.shape[0], :] = up_overlapped
        next_patch_modified[:, :left_overlapped.shape[1]] = left_overlapped
        next_patch_modified[:, -right_overlapped.shape[1]:] = right_overlapped
        next_patch_modified[-down_overlapped.shape[0]:, :] = down_overlapped

        return next_patch_modified


def texture_synthesis(img,
                      hole_top_left,
                      hole_shape,
                      hole_top_left2=None,
                      hole_shape2=None,
                      hole_top_left3=None,
                      hole_shape3=None,
                      patch_factor=0.9,
                      overlapping_factor=0.4,
                      max_allowable_movement=4):
    patch_shape = (int(patch_factor * hole_shape[0]), int(patch_factor * hole_shape[1]))
    v_overlap_shape = (patch_shape[0], int(patch_shape[1] * overlapping_factor))
    h_overlap_shape = (int(patch_shape[0] * overlapping_factor), patch_shape[1])
    # print(patch_shape)
    # print(v_overlap_shape)
    # print(h_overlap_shape)

    hole_shape[0] = hole_shape[0] - np.mod(hole_shape[0], patch_shape[0] - h_overlap_shape[0]) + \
                    patch_shape[0] - h_overlap_shape[0]
    hole_shape[1] = hole_shape[1] - np.mod(hole_shape[1], patch_shape[1] - v_overlap_shape[1]) + \
                    patch_shape[1] - v_overlap_shape[1]

    img[hole_top_left[0]:hole_top_left[0] + hole_shape[0], hole_top_left[1]:hole_top_left[1] + hole_shape[1]] = 0
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    number_of_rows = hole_shape[0] // (patch_shape[0] - h_overlap_shape[0])
    number_of_cols = hole_shape[1] // (patch_shape[1] - v_overlap_shape[1])
    # Filling the hole
    for i in range(number_of_rows):
        for j in range(number_of_cols):
            # x: axis 0
            # y: axis 1
            x_st_next = hole_top_left[0] - h_overlap_shape[0] + i * (patch_shape[0] - h_overlap_shape[0])
            y_st_next = hole_top_left[1] - v_overlap_shape[1] + j * (patch_shape[1] - v_overlap_shape[1])
            up_overlapping = img[x_st_next:x_st_next + h_overlap_shape[0], y_st_next:y_st_next + h_overlap_shape[1]]
            left_overlapping = img[x_st_next:x_st_next + v_overlap_shape[0], y_st_next:y_st_next + v_overlap_shape[1]]
            next_patch = find_matching_patch_L_shape(img,
                                                     up_overlapping,
                                                     left_overlapping,
                                                     patch_shape,
                                                     hole_top_left,
                                                     hole_shape,
                                                     hole_top_left2,
                                                     hole_shape2,
                                                     hole_top_left3,
                                                     hole_shape3,
                                                     )
            next_patch = merging(left_overlapping, up_overlapping, next_patch, 'L_shape_cut', max_allowable_movement)
            img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]] = next_patch.copy()
            # plt.figure()
            # plt.imshow(img)
            # plt.show()
    # Correcting the patches on right edge of the hole
    for i in range(number_of_rows):
        x_st_next = hole_top_left[0] - h_overlap_shape[0] + i * (patch_shape[0] - h_overlap_shape[0])
        y_st_next = hole_top_left[1] - v_overlap_shape[1] + number_of_cols * (patch_shape[1] - v_overlap_shape[1])
        patch_temp = img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]]
        overlapping_left = patch_temp[:, :v_overlap_shape[1]]
        overlapping_right = patch_temp[:, -v_overlap_shape[1]:]
        overlapping_up = patch_temp[:h_overlap_shape[0]:, :]
        temp = find_matching_patch_U_shape(img,
                                           overlapping_left,
                                           overlapping_up,
                                           overlapping_right,
                                           patch_shape,
                                           [x_st_next, y_st_next],
                                           hole_top_left,
                                           hole_shape,
                                           hole_top_left2,
                                           hole_shape2,
                                           hole_top_left3,
                                           hole_shape3,
                                           )
        temp = merging(overlapping_left, overlapping_up, temp, 'U_shape_cut', max_allowable_movement, overlapping_right)
        img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]] = temp.copy()
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
    # Correcting the patches on the down edge of the hole
    for j in range(number_of_cols):
        x_st_next = hole_top_left[0] - h_overlap_shape[0] + number_of_rows * (patch_shape[0] - h_overlap_shape[0])
        y_st_next = hole_top_left[1] - v_overlap_shape[1] + j * (patch_shape[1] - v_overlap_shape[1])
        patch_temp = img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]]
        overlapping_left = patch_temp[:, :v_overlap_shape[1]]
        overlapping_down = patch_temp[-h_overlap_shape[0]:, :]
        overlapping_up = patch_temp[:h_overlap_shape[0]:, :]
        temp = find_matching_patch_C_shape(img, overlapping_left,
                                           overlapping_up,
                                           overlapping_down,
                                           patch_shape,
                                           [x_st_next, y_st_next],
                                           hole_top_left,
                                           hole_shape,
                                           hole_top_left2,
                                           hole_shape2,
                                           hole_top_left3,
                                           hole_shape3,
                                           )
        temp = merging(overlapping_left, overlapping_up, temp, 'C_shape_cut', max_allowable_movement, overlapping_down)
        img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]] = temp.copy()
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

    # Correcting the patch on the right down corner of the hole
    x_st_next = hole_top_left[0] - h_overlap_shape[0] + number_of_rows * (patch_shape[0] - h_overlap_shape[0])
    y_st_next = hole_top_left[1] - v_overlap_shape[1] + number_of_cols * (patch_shape[1] - v_overlap_shape[1])
    patch_temp = img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]]
    overlapping_left = patch_temp[:, :v_overlap_shape[1]]
    overlapping_right = patch_temp[:, -v_overlap_shape[1]:]
    overlapping_down = patch_temp[-h_overlap_shape[0]:, :]
    overlapping_up = patch_temp[:h_overlap_shape[0]:, :]
    temp = find_matching_patch_O_shape(img,
                                       overlapping_left,
                                       overlapping_up,
                                       overlapping_right,
                                       overlapping_down,
                                       patch_shape,
                                       [x_st_next, y_st_next],
                                       hole_top_left,
                                       hole_shape,
                                       hole_top_left2,
                                       hole_shape2,
                                       hole_top_left3,
                                       hole_shape3,
                                       )
    temp = merging(overlapping_left,
                   overlapping_up,
                   temp,
                   'O_shape_cut',
                   max_allowable_movement,
                   overlapping_right,
                   overlapping_down)
    img[x_st_next:x_st_next + patch_shape[0], y_st_next:y_st_next + patch_shape[1]] = temp.copy()
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    return img


image = cv2.imread('im04.jpg', cv2.IMREAD_UNCHANGED)
if image is None:
    raise Exception("Couldn't load the image")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
top_left1 = (672, 722)
bottom_right1 = (1169, 945)
hole_shape_1 = [bottom_right1[0] - top_left1[0], bottom_right1[1] - top_left1[1]]
filled1 = texture_synthesis(image, top_left1, hole_shape_1)
plt.imsave('res16.jpg', filled1)

image2 = cv2.imread('im03.jpg', cv2.IMREAD_UNCHANGED)
if image2 is None:
    raise Exception("Couldn't load the image")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
top_left2 = (612, 1129)
bottom_right2 = (728, 1228)
top_left3 = (735, 823)
bottom_right3 = (874, 970)
top_left4 = (60, 319)
bottom_right4 = (171, 542)
hole_shape_2 = [bottom_right2[0] - top_left2[0], bottom_right2[1] - top_left2[1]]
hole_shape_3 = [bottom_right3[0] - top_left3[0], bottom_right3[1] - top_left3[1]]
hole_shape_4 = [bottom_right4[0] - top_left4[0], bottom_right4[1] - top_left4[1]]
# plt.figure()
# plt.imshow(image2)
# plt.show()

image2 = texture_synthesis(image2.copy(), top_left2, hole_shape_2, top_left3, hole_shape_3, top_left4, hole_shape_4)
image2 = texture_synthesis(image2.copy(), top_left3, hole_shape_3, top_left4, hole_shape_4)
filled2 = texture_synthesis(image2.copy(), top_left4, hole_shape_4, patch_factor=0.6, overlapping_factor=0.25)


plt.imsave('res15.jpg', filled2)
# plt.figure()
# plt.imshow(filled4)
# plt.show()