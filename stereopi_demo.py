import numpy as np
import cv2


def wrap_position_calculation(frame, position_calculation_method):
    SCALE = 5

    CROP_SHIFT = 30

    (YSZ, XSZ, _) = frame.shape
    XSZ = int(XSZ / 2)

    resized = cv2.resize(frame, (int(XSZ * 2 / SCALE), int(YSZ / SCALE)))

    XSZ = int(XSZ / SCALE)
    YSZ = int(YSZ / SCALE)

    left = resized[:, : XSZ, :]
    right = resized[:, XSZ:, :]

    left_crop = left[CROP_SHIFT: - CROP_SHIFT, CROP_SHIFT: - CROP_SHIFT, :]
    right_crop = right[CROP_SHIFT: - CROP_SHIFT, CROP_SHIFT: - CROP_SHIFT, :]

    position_calculation_method(left_crop, right_crop)


def loop_video(video_path, position_calculation_method):
    cam = cv2.VideoCapture(video_path)

    while (True):
        _, frame = cam.read()

        k = cv2.waitKey(10) & 0xFF

        if (frame is None):
            cam.release()
            cam = cv2.VideoCapture(video_path)

        wrap_position_calculation(frame, position_calculation_method)

        if (k == ord('q')):
            print("exiting")
            break

    cam.release()
    cv2.destroyAllWindows()


def find_max_bounding_box(mask, bbox_num=1):
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    stats = output[2]

    sorted_components = stats[np.argsort(stats[:, cv2.CC_STAT_AREA])]
    sorted_components = sorted_components[: -1]
    sorted_components = sorted_components[- min(
        bbox_num, len(sorted_components)):]

    result = []

    for i in range(len(sorted_components)):
        top = sorted_components[i, cv2.CC_STAT_TOP]
        left = sorted_components[i, cv2.CC_STAT_LEFT]
        width = sorted_components[i, cv2.CC_STAT_WIDTH]
        height = sorted_components[i, cv2.CC_STAT_HEIGHT]

        result.append(((left, top), (left + width, top + height)))

    return result


backgr_subtr_l = cv2.createBackgroundSubtractorMOG2()
backgr_subtr_r = cv2.createBackgroundSubtractorMOG2()


def calc_z_by_disparity(left, right):
    global backgr_subtr_l
    global backgr_subtr_r

    OBJ_LTH = (150, 150, 150)
    OBJ_HTH = (255, 255, 255)

    foreground_mask_l = backgr_subtr_l.apply(left)
    foreground_mask_r = backgr_subtr_r.apply(right)

    color_mask_l = cv2.inRange(left, OBJ_LTH, OBJ_HTH)
    color_mask_r = cv2.inRange(right, OBJ_LTH, OBJ_HTH)

    mask_l = cv2.bitwise_and(foreground_mask_l, color_mask_l)
    mask_r = cv2.bitwise_and(foreground_mask_r, color_mask_r)

    mask_l_bgr = cv2.cvtColor(mask_l, cv2.COLOR_GRAY2BGR)
    mask_r_bgr = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)

    left_row = np.concatenate((left,
                               cv2.cvtColor(foreground_mask_l, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(color_mask_l, cv2.COLOR_GRAY2BGR),
                               mask_l_bgr), axis=1)

    right_row = np.concatenate((right,
                                cv2.cvtColor(foreground_mask_r, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(color_mask_r, cv2.COLOR_GRAY2BGR),
                                mask_r_bgr), axis=1)

    # extract biggest connected component, draw bboxes, stack marked
    # images into object_row (+pad with copies)

    l_bboxes = find_max_bounding_box(mask_l)
    r_bboxes = find_max_bounding_box(mask_r)

    if (len(l_bboxes) == 0 or len(r_bboxes) == 0):
        print("skipping frame: no object")
        return

    l_bbox = l_bboxes[0]
    r_bbox = r_bboxes[0]

    mask_l_marked = cv2.rectangle(mask_l_bgr, l_bbox[0], l_bbox[1], (100, 200, 10), 3)
    mask_r_marked = cv2.rectangle(mask_r_bgr, r_bbox[0], r_bbox[1], (100, 200, 10), 3)

    object_row = np.concatenate((mask_l_marked, mask_r_marked, mask_r_marked, mask_r_marked),
                                axis=1)

    # concatenate left_row, right_row, object_row along 0 axis
    result = np.concatenate((left_row, right_row, object_row), axis=0)

    # subtract bbox middle points' x (disparity)
    disparity = (l_bbox[0][0] + l_bbox[1][0]) / 2 - (r_bbox[0][0] + r_bbox[1][0]) / 2

    # 1 / disparity * scale \approx distance
    not_distance = 1.0 / disparity

    # cv2. write distance over the frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(result, 'not distance: ' + str(not_distance), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    # cv2.imshow ("left", left_row)
    # cv2.imshow ("right", right_row)

    cv2.imshow("stages", result)


# def calc_z_by_point_cloud (left, right):


video_path = "/Users/elijah/Documents/stereopi/video.avi"

loop_video(video_path, calc_z_by_disparity)