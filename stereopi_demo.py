import numpy as np
import cv2
import picamera
from picamera import PiCamera
import time
import os
from datetime import datetime

def undistort(image, mapping):
    #w = 320
    #h = 240
    #print("Undistorting picture with (width, height):", (w, h))

    #print(image.shape)

    if 'map1' and 'map2' in mapping.files:
        #print("Camera calibration data has been found in cache.")
        map1 = mapping['map1']
        map2 = mapping['map2']
    else:
        print("Camera data file found but data corrupted.")
        exit(0)
    
    #except:
    #    print("Camera calibration data not found in cache, file " & './calibration_data/{}p/camera_calibration{}.npz'.format(h, left))
    #    exit(0)

    # We didn't load a new image from file, but use last image loaded while calibration
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted

def undistort_stereo(image, mapping):

    # We didn't load a new image from file, but use last image loaded while calibration
    undistorted = cv2.remap(image, mapping[0], mapping[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted

def wrap_position_calculation(frame, position_calculation_method, l_mapping, r_mapping):
    SCALE = 1

    CROP_SHIFT = 0

    (YSZ, XSZ, _) = frame.shape
    XSZ = int(XSZ / 2)

    left = frame[:, : XSZ, : 3]
    right = frame[:, XSZ:, : 3]

    if (CROP_SHIFT > 0):
        left_crop = left[CROP_SHIFT: - CROP_SHIFT, CROP_SHIFT: - CROP_SHIFT, :]
        right_crop = right[CROP_SHIFT: - CROP_SHIFT, CROP_SHIFT: - CROP_SHIFT, :]
    
    else:
        left_crop  = left
        right_crop = right

    left_undistorted  = undistort_stereo(left_crop,  l_mapping)
    right_undistorted = undistort_stereo(right_crop, r_mapping)
    
    cv2.imshow("undistorted", np.concatenate((left_undistorted, right_undistorted), axis=1))

    l_un_resized = cv2.resize(left_undistorted, (int(XSZ / SCALE), int(YSZ / SCALE)))
    r_un_resized = cv2.resize(right_undistorted, (int(XSZ / SCALE), int(YSZ / SCALE)))
    
    position_calculation_method(left_crop, right_crop, l_un_resized, r_un_resized)


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


def nothing(x):
    pass

def read_from_cameras(position_calculation_method):
    filename = './scenes/photo.png'

    win_name = "stages"
    cv2.namedWindow(win_name)
    cv2.createTrackbar("rl", "stages", 0, 255, nothing)
    cv2.createTrackbar("rh", "stages", 27, 255, nothing)
    cv2.createTrackbar("gl", "stages", 0, 255, nothing)
    cv2.createTrackbar("gh", "stages", 27, 255, nothing)
    cv2.createTrackbar("bl", "stages", 44, 255, nothing)
    cv2.createTrackbar("bh", "stages", 90, 255, nothing)
    
    cv2.createTrackbar("coeff", "stages", 100, 1000, nothing)

    # Camera settimgs
    cam_width = 1280
    cam_height = 480

    #load stereopair params
    cam_par_l = np.load('/home/pi/stereopi-fisheye-robot/calibration_data/{}p/camera_calibration{}.npz'.format(cam_height, '_left'))
    cam_par_r = np.load('/home/pi/stereopi-fisheye-robot/calibration_data/{}p/camera_calibration{}.npz'.format(cam_height, '_right'))
    print(cam_par_l["camera_matrix"])
    print(cam_par_r["camera_matrix"])
    npzfile = np.load('/home/pi/stereopi-fisheye-robot/calibration_data/{}p/stereo_camera_calibration.npz'.format(cam_height))
    cam_par_l = (npzfile['leftMapX'], npzfile['leftMapY'])
    cam_par_r = (npzfile['rightMapX'], npzfile['rightMapY'])
    print(npzfile["dispartityToDepthMap"])
    # Final image capture settings
    scale_ratio = 1

    # Camera resolution height must be dividable by 16, and width by 32
    cam_width = int((cam_width+31)/32)*32
    cam_height = int((cam_height+15)/16)*16
    print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

    # Buffer for captured image settings
    img_width = int (cam_width * scale_ratio)
    img_height = int (cam_height * scale_ratio)
    capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

    # Initialize the camera
    camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
    camera.resolution=(cam_width, cam_height)
    camera.framerate = 20
    camera.hflip = True

    t2 = datetime.now()
    counter = 0
    avgtime = 0
    # Capture frames from the camera
    for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
        #print("after reading", frame.shape)
        
        counter+=1
        t1 = datetime.now()
        timediff = t1-t2
        avgtime = avgtime + (timediff.total_seconds())
        #cv2.imshow("pair", frame)
        wrap_position_calculation(frame, position_calculation_method, cam_par_l, cam_par_r)
        
        key = cv2.waitKey(1) & 0xFF
        t2 = datetime.now()
        # if the `q` key was pressed, break from the loop and save last image
        if key == ord("q") :
            avgtime = avgtime/counter
            print ("Average time between frames: " + str(avgtime))
            print ("Average FPS: " + str(1/avgtime))
            if (os.path.isdir("./scenes")==False):
                os.makedirs("./scenes")
            cv2.imwrite(filename, frame)
            exit(0)
            break
   
    

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


def calc_z_by_disparity(l_crop, r_crop, left, right):
    rl = cv2.getTrackbarPos("rl", "stages")
    rh = cv2.getTrackbarPos("rh", "stages")
    gl = cv2.getTrackbarPos("gl", "stages")
    gh = cv2.getTrackbarPos("gh", "stages")
    bl = cv2.getTrackbarPos("bl", "stages")
    bh = cv2.getTrackbarPos("bh", "stages")

    coeff = cv2.getTrackbarPos("coeff", "stages")

    OBJ_LTH = (rl, gl, bl)
    OBJ_HTH = (rh, gh, bh)

    #print("shape", left.shape)

    color_mask_l = cv2.inRange(left, OBJ_LTH, OBJ_HTH)
    color_mask_r = cv2.inRange(right, OBJ_LTH, OBJ_HTH)

    #mask_l = cv2.bitwise_and(foreground_mask_l, color_mask_l)
    #mask_r = cv2.bitwise_and(foreground_mask_r, color_mask_r)

    mask_l_bgr = cv2.cvtColor(color_mask_l, cv2.COLOR_GRAY2BGR)
    mask_r_bgr = cv2.cvtColor(color_mask_r, cv2.COLOR_GRAY2BGR)

    left_row  = np.concatenate((l_crop, left,   mask_l_bgr), axis=1)
    right_row = np.concatenate((r_crop, right, mask_r_bgr), axis=1)

    # extract biggest connected component, draw bboxes, stack marked
    # images into object_row (+pad with copies)

    l_bboxes = find_max_bounding_box(color_mask_l)
    r_bboxes = find_max_bounding_box(color_mask_r)

    if (len(l_bboxes) == 0 or len(r_bboxes) == 0):
        #print("skipping frame: no object")
        return

    l_bbox = l_bboxes[0]
    r_bbox = r_bboxes[0]

    mask_l_marked = cv2.rectangle(mask_l_bgr, l_bbox[0], l_bbox[1], (100, 200, 10), 3)
    mask_r_marked = cv2.rectangle(mask_r_bgr, r_bbox[0], r_bbox[1], (100, 200, 10), 3)

    object_row = np.concatenate((mask_l_marked, mask_r_marked,
                                 mask_r_marked), axis=1)

    # concatenate left_row, right_row, object_row along 0 axis
    result = np.concatenate((left_row, right_row, object_row), axis=0)

    # subtract bbox middle points' x (disparity)
    #disparity = l_bbox[0][0] + l_bbox[1][0] / 2 - r_bbox[0][0] - r_bbox[1][0] / 2
    
    #print(l_bbox, r_bbox)
    
    disparity = np.abs(l_bbox[0][0] - r_bbox[0][0]) - 71

    # 1 / disparity * scale \approx distance
    not_distance = 100.0 * coeff / disparity

    # cv2. write distance over the frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(result, 'distance: ' + str(not_distance) [:8] + " / " + str(disparity), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    # cv2.imshow ("left", left_row)
    # cv2.imshow ("right", right_row)

    scale = 3
    (h, w, _) = result.shape
    resized = cv2.resize(result, (int(w / scale), int(h / scale)))

    cv2.imshow("stages", resized)


# def calc_z_by_point_cloud (left, right):


video_path = "/Users/elijah/Documents/stereopi/video.avi"

#loop_video(video_path, calc_z_by_disparity)

read_from_cameras(calc_z_by_disparity)
