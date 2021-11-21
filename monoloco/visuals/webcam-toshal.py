import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import requests
import math
import time
import itertools 
import csv
import datetime

import torchvision
from torchvision import transforms

from contextlib import contextmanager
import math
from ..visuals import Printer
from ..visuals.pifpaf_show import KeypointPainter, image_canvas
from ..network import PifPaf, MonoLoco
from ..network.process import preprocess_pifpaf, factory_for_gt, image_transform
from matplotlib.patches import Circle, FancyArrow
from vidgear.gears import WriteGear


from ..tracking_utils.utils import mkdir_if_missing
from ..lib.multitracker import JDETracker
from ..tracking_utils.timer import Timer

from google.cloud import logging
from google.cloud import storage



def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou, boxB

def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular

    return img, ratio, dw, dh

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_filename(source_file_name)    
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    blob.make_public()    
    print(blob.public_url)
    return blob.public_url

def save_to_video(buffer):
    file_name = str((time.time()))
    file_name = file_name.split(".")[0] + file_name.split(".")[1] + ".mp4"
    output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}
    writer = WriteGear(output_filename = file_name, logging = True, **output_params)

    for frame in buffer:
        writer.write(frame)
    
    writer.close() 

    return file_name


def webcam(args):
    property_id = 1
    floor1cams = [1,2,3,4,5,7,8,9,10]

    logging_client = logging.Client()
    log_name = 'python-algo-log'
    logger = logging_client.logger(log_name)

    logging_interval = 0
    csv_counter = 0
    norm_counter = 0

    tracker = JDETracker("Muna", frame_rate=15)

    camera_ip_address = []
    camera_desp = []
    camera_ids = []
    results = []

    person_hash = {}
    person_timeslice = {} #key - person id, values [first_instace, last instance]

    person_incident_hash = {}
    person_incident_timeslice = {}

    incident_group = 0
    buffer = []
    unique_reid = []

    group_count = {}
    groups = {}

    HEADER = {
        "content-type":"application/json",
        "Authorization": "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzUsImlhdCI6MTYzMzA2OTYwMn0.cyhiM5Gr_x3NB7vNfEHQtXv5OU19miV_prkU13_LDlY",
        #"x-api-key":"6cf06df6bb9d4bee82c6a965c166c973"
    }
    IP_ADDRESS = 'localhost:7000'
    INCIDENT_LOG_ENDPOINT = 'http://localhost:7000/api/incident-logs'
    PERSON_INSTANCE_ENDPOINT = 'http://localhost:7000/api/persons/person-instances/'
    PERSON_ENDPOINT = 'http://localhost:7000/api/persons/'
    LOGIN_API = 'http://localhost:7000/auth/login'
    PERSON_PERSON_INSTANCE_ENDPOINT = 'http://localhost:7000/api/person-person-instances'
    PERSON_PER_TIMESLICE_ENDPOINT = 'http://localhost:7000/api/persons/person-per-timeslice'
    INCIDENT_ENDPOINT = 'http://localhost:7000/api/incidents'
    INCIDENT_PERSON_INSTANCE_ENDPOINT = 'http://localhost:7000/api/incidents/person-instance'

    try:
        PARAMS = {'password': int(1234), 'email': 'admin@lauretta.io'}
        r = requests.post(url = LOGIN_API, json = PARAMS, headers=HEADER)
        print(r)

    except Exception as e:
        print(e)
        text = 'Camera API Exception, Cant read Camera API'
        logger.log_text(text)
        print('Logged: {}'.format(text))

    startup_var = 0
    counter = 0
    min_box_area = 100

    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')

    trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt'))
    model_fair_7 = model_fair_7.to(args.device)
    model_fair_7.eval()

    args.camera = True
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.model, device=args.device)

    camera_api = "http://" + str(IP_ADDRESS)+ "/api/cameras"

    r = requests.get(camera_api, headers=HEADER)
    print(r.json())

    for i in (r.json()['data']):
        camera_ip_address.append(i['camera_ip_address'])
        camera_desp.append(i['description'])
        camera_ids.append(i['id'])

    print(camera_ids)
    print(camera_desp)
    print(camera_ip_address)
    #cv2.namedWindow('Remote', cv2.WINDOW_AUTOSIZE)

    for camera_addr in itertools.cycle(camera_ip_address):
        index = camera_ip_address.index(camera_addr)
        try:
            cam = cv2.VideoCapture(camera_addr)
        except Exception as e:
            continue
        max_incidents = []
        max_camera = None

        #cam =  VideoCaptureThreading(camera_addr).start()
        prev_time = time.time()
        while True:
            cur_time = time.time()
            if cur_time - prev_time < 4:
                try:
                    ret, frame = cam.read()
                    img0 = frame.copy()
                except Exception:
                    continue
                cv2.imshow('Remote', frame)
                cv2.waitKey(1)
                image = frame.copy()
                imagex = frame.copy()

                if len(buffer) < 50:
                    buffer.append(frame)
                else:
                    buffer = []
                    buffer.append(frame)
                height, width, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image_cpu = image_transform(image.copy())
                processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
                fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
                _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

                pil_image = Image.fromarray(image)
                intrinsic_size = [xx * 1.3 for xx in pil_image.size]
                kk, dict_gt = factory_for_gt(intrinsic_size)  # better intrinsics for mac camera
                

                
                
                ##################RE-ID###############################################
                #img0 = cv2.resize(img0, (w, h))
                img, _, _, _ = letterbox(img0, height=608, width=1088)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                online_targets = tracker.update(blob, img0)

                online_tlbr = []
                online_ids = []
                #online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    
                    x1_reid = int(tlwh[0])
                    y1_reid = int(tlwh[1])
                    x2_reid = int(tlwh[0] + tlwh[2])
                    y2_reid = int(tlwh[1] + tlwh[3])

                    if 1:
                        online_tlbr.append([x1_reid, y1_reid, x2_reid, y2_reid])
                        online_ids.append(tid)

                #########################################################################
                if pifpaf_out:
                    boxes, keypoints = preprocess_pifpaf(pifpaf_out, (width, height))
                    dic_out = monoloco.forward(keypoints, kk)
                    dic_out = monoloco.post_process(dic_out, boxes, keypoints, kk, dict_gt, reorder=False)
                    #show_social(args, image, str(counter), pifpaf_out, dic_out)
                    violations, non_violations = record_social(args, image, str(counter), pifpaf_out, dic_out, camera_desp[index], camera_ids[index], INCIDENT_LOG_ENDPOINT, PERSON_INSTANCE_ENDPOINT, model_fair_7, trans)
                    max_incidents.append(len(violations))

                    for ilv in range(0, len(non_violations)):
                        max_iou = 0
                        max_iou_index = -1
                        age_temp = None
                        race_temp = None
                        gender_temp = None
                        monoloco_idx, camera_id,floor_id,incident_type_id,box_x1, box_y1, box_x2, box_y2, floor_x1, floor_y1, floor_x2, floor_y2, status_id, age, gender, race, direction_degrees = non_violations[ilv]
                        
                        try:
                            PARAMS = {'property_id':int(property_id),'cam_id': int(camera_id), 'floor_id': int(floor_id), 'person_instance_x1': box_x1, 'person_instance_y1': box_y1, 'person_instance_x2': box_x2, 'person_instance_y2': box_y2, 'floorplan_x1': floor_x1, 'floorplan_y1': floor_y1, 'floorplan_x2': floor_x2, 'floorplan_y2' : floor_y2}
                            print(PARAMS)
                            print(HEADER)
                            q = requests.post(url = PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                            person_instance_id = q.json()['data']['id']

                        except Exception as e:
                            text = 'Person Instance API POST Problem, Cant Write'
                            logger.log_text(text)
                            print('Logged: {}'.format(text))
                            continue

                        for ilw in range(0, len(online_tlbr)):
                            iou, reid_xyxy = bb_intersection_over_union([box_x1,box_y1,box_x2,box_y2], online_tlbr[ilw])
                            if iou > max_iou:
                                max_iou = iou
                                max_iou_index = ilw

                        if max_iou > 0.30:
                            for idx_val in range(0, len(online_ids)):
                                if online_ids[idx_val] not in groups.keys() and not groups:
                                    groups[online_ids[idx_val]] = online_ids
                                elif online_ids[idx_val] in groups.keys():
                                    if online_ids[idx_val] not in group_count.keys():
                                        group_count[online_ids[idx_val]] = 1
                                    else:
                                        group_count[online_ids[idx_val]] = group_count[online_ids[idx_val]] + 1
                                elif online_ids[idx_val] not in groups.keys():
                                    temp = True
                                    for groups_idx in groups.values():
                                        if online_ids[idx_val] in groups_idx:
                                            temp = False
                                            if online_ids[idx_val] not in group_count.keys():
                                                group_count[online_ids[idx_val]] = 1
                                            else:
                                                group_count[online_ids[idx_val]] = group_count[online_ids[idx_val]] + 1
                                    if temp:
                                        groups[online_ids[idx_val]] = online_ids
                            unique_id = online_ids[max_iou_index]
                            if unique_id not in unique_reid:
                                unique_reid.append(unique_id)
                                try:
                                    # cv2.imwrite("/home/lauretta/lauretta_server_python/social-dist-1/person_imgs/" 
                                    # + str(unique_id) + ".jpg", imagex[int(box_y1):int(box_y2), int(box_x1):int(box_x2)])
                                    if age is not None:
                                        age_temp = age
                                    else:
                                        age_temp = 99

                                    if gender is not None:
                                        if gender == 0:
                                            gender_temp = 'Male'
                                        elif gender == 1:
                                            gender_temp = 'Female'
                                    else:
                                        gender_temp = 'NULL'

                                    if race is not None:
                                        if race == 0 or race == 6:
                                            race_temp = 'Chinese'
                                        elif race == 1 or race == 5:
                                            race_temp = 'Malay'
                                        elif race == 2 or race == 4:
                                            race_temp = 'Caucasian'
                                        else:
                                            race_temp = 'Indian'
                                    else:
                                        race_temp = 'NULL'


                                    PARAMS = {'property_id':int(property_id), 'race': race_temp, 'age': int(age_temp), 'gender': gender_temp}
                                    r = requests.post(url = PERSON_ENDPOINT, json = PARAMS, headers=HEADER)
                                    person_id = (r.json()['data']['id'])
                                    person_hash[unique_id] = person_id
                                    person_timeslice[unique_id] = datetime.datetime.now()

                                    try:
                                        PARAMS = {'property_id':int(property_id), 'person_id': int(person_id), 'person_instance_id': int(person_instance_id)}
                                        print("PERSON-PERSON inst: \n",PARAMS)
                                        r = requests.post(url = PERSON_PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                                    except Exception as e:
                                        print("Logging: person_instance not written ", PARAMS)
                                        continue
                                    if logging_interval % 10 == 0:
                                        text = 'Id: ' + str(unique_id) + ' Race: ' + str(race_temp) + ' Age: ' + str(age_temp) + ' Gender: ' + str(gender_temp)
                                        logger.log_text(text)
                                        print('Logged: {}'.format(text))
                                except Exception as e:
                                    text = 'Person API POST Problem, Cant Write'
                                    logger.log_text(text)
                                    print('Logged: {}'.format(text))
                                    continue
                                    
                            elif unique_id in unique_reid:
                                if unique_id in person_hash.keys():
                                    person_temp_id = person_hash[unique_id]
                                else:
                                    continue

                                timeslice_start_time = person_timeslice[unique_id]
                                timeslice_now = datetime.datetime.now()

                                if (((timeslice_now - timeslice_start_time).total_seconds()) >= 300):
                                    timeslice_start_time = datetime.datetime.now()
                                    interval = (int(timeslice_now.strftime("%H")) * 60) + int(timeslice_now.strftime("%M"))
                                    interval = math.floor(interval / 5)
                                    interval_id = str(timeslice_now.strftime("%Y%m%d")) + "_" + str(interval)
                                    try:
                                        PARAMS = {'property_id':int(property_id) ,'intervalID': int(interval_id), 'timestamp': timeslice_now, 'PersonID': int(person_temp_id), 'iou': '1', 'floorplan_x1': floor_x1, 'floorplan_y1':floor_y1, 'floor_id': int(floor_id), 'camera_id': int(camera_id)}
                                        r = requests.post(url = PERSON_PER_TIMESLICE_ENDPOINT, json = PARAMS, headers=HEADER)
                                    except Exception as e:
                                        print("LOGGER: PERSON_PER_TIMESLICE not written\n", PARAMS)
                                        continue
                                
                                try:
                                    PARAMS = {'property_id':int(property_id), 'person_id': int(person_temp_id), 'person_instance_id': int(person_instance_id)}
                                    print("PERSON-PERSON inst: \n",PARAMS)
                                    r = requests.post(url = PERSON_PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                                except Exception as e:
                                    print("Logger: Person Person Instance not written, ",PARAMS)
                                    continue

                    ## NON
                    person_buffer = []
                    for ilv in range(0, len(violations)):
                        max_iou = 0
                        max_iou_index = -1
                        age_temp = None
                        race_temp = None
                        gender_temp = None
                        try:
                            person_buffer.append(unique_id)
                        except:
                            continue
                        distance = 1.3

                        monoloco_idx, camera_id,floor_id,incident_type_id,box_x1, box_y1, box_x2, box_y2, floor_x1, floor_y1, floor_x2, floor_y2, status_id, age, gender, race, direction_degrees, distance_count = violations[ilv]
                        max_camera = int(camera_id)
                        try:
                            curr_time = datetime.datetime.now().isoformat()
                            PARAMS = {
                                'property_id':int(property_id) ,'cam_id': int(camera_id), 'created_at':curr_time, 
                                'floor_id': int(floor_id), 'incident_type_id': incident_type_id, 'cam_view_x1': box_x1, 
                                'cam_view_y1': box_y1, 'cam_view_x2': box_x2, 'cam_view_y2': box_y2, 'floorplan_x1': floor_x1, 
                                'floorplan_y1': floor_y1, 'floorplan_x2' : floor_x2, 'floorplan_y2': floor_y2, 'total_people': len(violations),
                                'footage_path': '', 'status_id': status_id, 'person_ids': person_buffer, 'distance': distance
                            }
                            r = requests.post(url = INCIDENT_LOG_ENDPOINT, json = PARAMS, headers=HEADER)
                            print(r)

                        except Exception as e:
                            text = 'Incident Log POST Problem, Cant Write'
                            logger.log_text(text)
                            print('Logged: {}'.format(text))
                            continue

                        try:
                            PARAMS = {
                                'property_id':int(property_id),'cam_id': int(camera_id), 
                                'floor_id': int(floor_id), 'person_instance_x1': box_x1, 'person_instance_y1': box_y1, 
                                'person_instance_x2': box_x2, 'person_instance_y2': box_y2, 'floorplan_x1': floor_x1, 
                                'floorplan_y1': floor_y1, 'floorplan_x2': floor_x2, 'floorplan_y2' : floor_y2, 'direction': direction_degrees
                            }
                            print(PARAMS)
                            q = requests.post(url = PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                            person_instance_id = q.json()['data']['id']

                        except Exception as e:
                            text = 'Person Instance API POST Problem, Cant Write'
                            logger.log_text(text)
                            print('Logged:2 {}'.format(text))
                            continue

                        for ilw in range(0, len(online_tlbr)):
                            iou, reid_xyxy = bb_intersection_over_union([box_x1,box_y1,box_x2,box_y2], online_tlbr[ilw])
                            if iou > max_iou:
                                max_iou = iou
                                max_iou_index = ilw
                        if max_iou > 0.30: 
                            for idx_val in range(0, len(online_ids)):
                                if online_ids[idx_val] not in groups.keys() and not groups:
                                    groups[online_ids[idx_val]] = online_ids
                                elif online_ids[idx_val] in groups.keys():
                                    if online_ids[idx_val] not in group_count.keys():
                                        group_count[online_ids[idx_val]] = 1
                                    else:
                                        group_count[online_ids[idx_val]] = group_count[online_ids[idx_val]] + 1
                                elif online_ids[idx_val] not in groups.keys():
                                    temp = True
                                    for groups_idx in groups.values():
                                        if online_ids[idx_val] in groups_idx:
                                            temp = False
                                            if online_ids[idx_val] not in group_count.keys():
                                                group_count[online_ids[idx_val]] = 1
                                            else:
                                                group_count[online_ids[idx_val]] = group_count[online_ids[idx_val]] + 1
                                    if temp:
                                        groups[online_ids[idx_val]] = online_ids
                            unique_id = online_ids[max_iou_index]
                            if unique_id not in unique_reid:
                                unique_reid.append(unique_id)
                                try:
                                    cv2.imwrite("/home/lauretta/lauretta_server_python/social-dist-1/person_imgs/" + str(unique_id) +
                                     ".jpg", imagex[int(box_y1):int(box_y2), int(box_x1):int(box_x2)])    
                                    if age is not None:
                                        age_temp = age
                                    else:
                                        age_temp = 99

                                    if gender is not None:
                                        if gender == 0:
                                            gender_temp = 'Male'
                                        elif gender == 1:
                                            gender_temp = 'Female'
                                    else:
                                        gender_temp = 'NULL'

                                    if race is not None:
                                        if race == 0 or race == 6:
                                            race_temp = 'Chinese'
                                        elif race == 1 or race == 5:
                                            race_temp = 'Malay'
                                        elif race == 2 or race == 4:
                                            race_temp = 'Caucasian'
                                        else:
                                            race_temp = 'Indian'
                                    else:
                                        race_temp = 'NULL'

                                    
                                    PARAMS = {'property_id':int(property_id), 'race': race_temp, 'age': int(age_temp), 'gender': gender_temp}
                                    r = requests.post(url = PERSON_ENDPOINT, json = PARAMS, headers=HEADER)
                                    person_id = (r.json()['data']['id'])
                                    person_hash[unique_id] = person_id 
                                    person_timeslice[unique_id] = datetime.datetime.now()

                                    person_incident_hash[unique_id] = person_id
                                    person_incident_timeslice[unique_id] = datetime.datetime.now()

                                    try:
                                        PARAMS = {'property_id': int(property_id), 'person_id': int(person_id), 'person_instance_id': int(person_instance_id)}
                                        print("PERSON-PERSON inst: \n",PARAMS)
                                        r = requests.post(url = PERSON_PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                                    except Exception as e:
                                        print("Logger: Person_person_instance not written: ", PARAMS)
                                        continue
                                    if logging_interval % 10 == 0:
                                        text = 'Id: ' + str(unique_id) + ' Race: ' + str(race_temp) + ' Age: ' + str(age_temp) + ' Gender: ' + str(gender_temp)
                                        logger.log_text(text)
                                        print('Logged: {}'.format(text))

                                except Exception as e:
                                    text = 'Person API POST Problem, Cant Write'
                                    logger.log_text(text)
                                    print('Logged: {}'.format(text))
                                    continue
                            
                            elif unique_id in unique_reid:
                                if unique_id in person_hash.keys():
                                    person_temp_id = person_hash[unique_id]
                                else:
                                    continue

                                person_temp_id = person_hash[unique_id]

                                timeslice_start_time = person_timeslice[unique_id]
                                timeslice_now = datetime.datetime.now()
                                try:
                                    person_temp_incident_id = person_incident_hash[unique_id]
                                    timeslice_incident_start_time = person_incident_timeslice[unique_id]

                                    if (((timeslice_now - timeslice_start_time).total_seconds()) >= 600):
                                        timeslice_start_time = datetime.datetime.now()
                                        try:
                                            file_name = save_to_video(buffer)
                                            url_link = upload_blob(
                                                bucket_name = "lauretta_property_video",
                                                source_file_name= str(file_name),
                                                destination_blob_name= "sdc/" + str(file_name),
                                            )
                                            try:
                                                PARAMS = {'property_id':int(property_id),'start_time': datetime.datetime.now(), 'end_time': datetime.datetime.now(),
                                                          'incident_type_id': 5, 'video_link': url_link, 
                                                          'floor_plan x': floor_x1, 'floorplan_y': floor_y1, 'floor_id':int(floor_id)}
                                                print("INCIDENT ENDPOINT\n", PARAMS)
                                                r = requests.post(url = INCIDENT_ENDPOINT, json = PARAMS, headers=HEADER)
                                                print(r)
                                            except Exception as e:
                                                print("Logger: INCIDENT_ENDPOINT not written")
                                                print(e)
                                        except:
                                            continue
                                except Exception as e:
                                    print(e)
                                    #continue


                                if (((timeslice_now - timeslice_start_time).total_seconds()) >= 300):
                                    timeslice_start_time = datetime.datetime.now()
                                    interval = (int(timeslice_now.strftime("%H")) * 60) + int(timeslice_now.strftime("%M"))
                                    interval = math.floor(interval / 5)
                                    interval_id = str(timeslice_now.strftime("%Y%m%d")) + "_" + str(interval)
                                    try:
                                        PARAMS = {'property_id':int(property_id) ,'intervalID': interval_id, 'timestamp': timeslice_now, 'PersonID': person_temp_id, 'iou': '1', 'floorplan_x1': floor_x1, 
                                        'floorplan_y1':floor_y1, 'floor_id': int(floor_id), 'camera_id': int(camera_id), 'angle':direction_degrees, 'incident_group':incident_group, 
                                        'incident_numpeople': distance_count}
                                        r = requests.post(url = PERSON_PER_TIMESLICE_ENDPOINT, json = PARAMS, headers=HEADER)
                                    except Exception as e:
                                        print("LOGGER: PERSON_PER_TIMESLICE not written \n", PARAMS)
                                        continue
                                try:
                                    PARAMS = {'property_id':int(property_id),'person_id': int(person_temp_id), 'person_instance_id': int(person_instance_id)}
                                    print("PERSON-PERSON inst: \n",PARAMS)
                                    r = requests.post(url = PERSON_PERSON_INSTANCE_ENDPOINT, json = PARAMS, headers=HEADER)
                                except Exception as e:
                                    print("Logger: person person instance not written ", PARAMS)
                                    continue
                counter += 1

                
            else:
                break
        cam.release()

        f = open("1.csv", "a")
        csvwriter = csv.writer(f)
        if max_camera is not None:
            csvwriter.writerows([[str(time.ctime(time.time())), str(csv_counter + 1), str(max_camera), str(max(max_incidents))]])
        f.close()

        norm_counter = norm_counter + 1
        if norm_counter % 34 == 0:
            csv_counter += 1

        if csv_counter % 34 == 0:
            csv_counter = 0

        logging_interval = logging_interval + 1



        for peoplex in group_count.keys():
            try:
                db_peopleid = person_hash[peoplex]
                count = group_count[peoplex]
                if count > 10:
                    parms = {'property_id':int(property_id)}
                    r = requests.get(PERSON_ENDPOINT + str(db_peopleid) + '?property[id]='+property_id, headers=HEADER)
                    race = r.json()['data']['race']
                    gender = r.json()['data']['gender']
                    age = r.json()['data']['age']
                    age_group = r.json()['data']['age_group']
                    travel_in_group =  r.json()['data']['travel_in_group']

                    payload = {'property_id':int(property_id), 'race': race, 'gender': gender, 'age': age, 'age_group': age_group, 'travel_in_group':1}
                    r = requests.patch(PERSON_ENDPOINT + str(db_peopleid), json=payload, headers=HEADER)
                    print(r)

            except:
                continue


        print(group_count)

        print(person_hash)

def record_social(args, image_t, output_path, annotations, dic_out, camera_disp, camera_id, incident_api, person_instance_api, model_fair_7, trans):
    '''
    Returns monoloco_idx, camera_id,floor_id,incident_type_id,box_x1, box_y1, box_x2, box_y2, floorplan_x1, floorplan_y1, floorplan_x2, floorplan_y2, status_id
    '''
    angles = dic_out['angles']
    xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]
    bboxes = dic_out['boxes']
    uv_centers = dic_out['uv_heads']
    sizes = [abs(dic_out['uv_heads'][idx_temp][1] - uv_st[1]) / 1.5 for idx_temp, uv_st in enumerate(dic_out['uv_shoulders'])]
    radiuses = [s / 1.2 for s in sizes]

    violations = []
    non_violations = []

    total_people = len(bboxes)
    status_id = 1
    incident_type_id = 1

    footage_path = ''

    violations_clusters = []

    for idx, _ in enumerate(dic_out['xyz_pred']):
        sdist_bool, distance_count = social_distance(xz_centers, angles, idx)
        if sdist_bool :
            if len(bboxes) > 0:
                box_x1 = bboxes[idx][0]
                box_y1 = bboxes[idx][1]
                box_x2 = bboxes[idx][2]
                box_y2 = bboxes[idx][3]
                
                if int(camera_id) in [1,2,3,4,5,7,8,9,10]:
                    floor_id = 1
                else:
                    floor_id = 2
                
                age, gender, race = crop_head(args, image_t,uv_centers[idx],radiuses[idx], angles[idx], model_fair_7, trans)

                floor_x1, floor_y1, floor_x2, floor_y2 = get_floor_cords(xz_centers, angles[idx], idx)
                direction_degrees = math.degrees(angles[idx])
                violations.append([idx,int(camera_id),int(floor_id),int(incident_type_id),box_x1, box_y1, box_x2, box_y2, floor_x1, floor_y1, floor_x2, floor_y2, status_id, age, gender, race, direction_degrees, distance_count])

        else:
            if len(bboxes) == 0:
                continue

            box_x1 = bboxes[idx][0]
            box_y1 = bboxes[idx][1]
            box_x2 = bboxes[idx][2]
            box_y2 = bboxes[idx][3]

            if int(camera_id) in [1,2,3,4,5,7,8,9,10]:
                floor_id = 1
            else:
                floor_id = 2

            age, gender, race = crop_head(args, image_t,uv_centers[idx],radiuses[idx], angles[idx], model_fair_7, trans)
            floor_x1, floor_y1, floor_x2, floor_y2 = get_floor_cords(xz_centers, angles[idx], idx)
            direction_degrees = math.degrees(angles[idx])
            non_violations.append([idx,int(camera_id),int(floor_id),int(incident_type_id),box_x1, box_y1, box_x2, box_y2, floor_x1, floor_y1, floor_x2, floor_y2, status_id, age, gender, race, direction_degrees])
                
    return violations, non_violations

def crop_head(args,image_t,center,radius, theta, model_fair_7, trans):
    radius = int(radius + 20)
    x,y = center
    x,y = int(x), int(y)
    muna = image_t[y-radius:y+radius, x-radius:x+radius]
    height, width, channels = muna.shape
    if height > 0 and width > 0 and 1:
        muna = cv2.cvtColor(muna, cv2.COLOR_BGR2RGB)
        muna = trans(muna)
        muna = muna.view(1, 3, 224, 224)
        muna = muna.to(args.device)
        outputs = model_fair_7(muna)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        return age_pred, gender_pred, race_pred
    else:
        return None, None, None

def get_floor_cords(centers, angles, idx):
    length = 0.5
    x_arr = centers[idx][0]
    z_arr = centers[idx][1]
    delta_x = length * math.cos(angles)
    delta_z = - length * math.sin(angles)

    return x_arr, z_arr, delta_x , delta_z #For Focal Length 50


def show_social(args, image_t, output_path, annotations, dic_out):
    """Output frontal image with poses or combined with bird eye view"""

    assert 'front' in args.output_types or 'bird' in args.output_types, "outputs allowed: front and/or bird"

    angles = dic_out['angles']
    xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

    colors = ['r' if social_distance(xz_centers, angles, idx) else 'deepskyblue'
              for idx, _ in enumerate(dic_out['xyz_pred'])]

    if 'front' in args.output_types:
        # Prepare colors
        keypoint_sets, scores = get_pifpaf_outputs(annotations)
        uv_centers = dic_out['uv_heads']
        sizes = [abs(dic_out['uv_heads'][idx][1] - uv_s[1]) / 1.5 for idx, uv_s in enumerate(dic_out['uv_shoulders'])]

        keypoint_painter = KeypointPainter(show_box=False)

        with image_canvas(image_t,
                          output_path + '.front.png',
                          show=args.show,
                          fig_width=10,
                          dpi_factor=1.0) as ax:
            print("Muna")
            crop_head(image_t, uv_centers, angles, sizes)
    if 'bird' in args.output_types:
        with bird_canvas(args, output_path) as ax1:
            draw_orientation(ax1, xz_centers, [], angles, colors, mode='bird')

def draw_orientation(ax, centers, sizes, angles, colors, mode):

    if mode == 'front':
        length = 5
        fill = False
        alpha = 0.6
        zorder_circle = 0.5
        zorder_arrow = 5
        linewidth = 1.5
        edgecolor = 'k'
        radiuses = [s / 1.2 for s in sizes]
    else:
        length = 1.3
        head_width = 0.3
        linewidth = 2
        radiuses = [0.2] * len(centers)
        fill = True
        alpha = 1
        zorder_circle = 2
        zorder_arrow = 1

    for idx, theta in enumerate(angles):
        color = colors[idx]
        radius = radiuses[idx] + 20

        if mode == 'front':
            x_arr = centers[idx][0] + (length + radius) * math.cos(theta)
            z_arr = length + centers[idx][1] + (length + radius) * math.sin(theta)
            delta_x = math.cos(theta)
            delta_z = math.sin(theta)
            head_width = max(10, radiuses[idx] / 1.5)

        else:
            edgecolor = color
            x_arr = centers[idx][0]
            z_arr = centers[idx][1]
            delta_x = length * math.cos(theta)
            delta_z = - length * math.sin(theta)  # keep into account kitti convention

        circle = Circle(centers[idx], radius=radius, color=color, fill=fill, alpha=alpha, zorder=zorder_circle)
        arrow = FancyArrow(x_arr, z_arr, delta_x, delta_z, head_width=head_width, edgecolor=edgecolor,
                           facecolor=color, linewidth=linewidth, zorder=zorder_arrow)
        ax.add_patch(circle)
        ax.add_patch(arrow)


def social_distance(centers, angles, idx, threshold=2.5):
    """
    return flag of alert if social distancing is violated
    """
    xx = centers[idx][0]
    zz = centers[idx][1]
    angle = angles[idx]
    distances = []
    distance_count = 0
    for i, _ in enumerate(centers):
        distance_temp = math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2)
        if distance_temp < threshold:
            distance_count = distance_count + 1
        distances.append(distance_temp)

    #distances = [math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2) for i, _ in enumerate(centers)]
    sorted_idxs = np.argsort(distances)

    #print(distances)

    for i in sorted_idxs[1:]:

        # First check for distance
        if distances[i] > threshold:
            return False, distance_count

        # More accurate check based on orientation and future position
        elif check_social_distance((xx, centers[i][0]), (zz, centers[i][1]), (angle, angles[i])):
            return True, distance_count

    return False, distance_count


def check_social_distance(xxs, zzs, angles):
    """
    Violation if same angle or ine in front of the other
    Obtained by assuming straight line, constant velocity and discretizing trajectories
    """
    min_distance = 0.5
    theta0 = angles[0]
    theta1 = angles[1]
    steps = np.linspace(0, 2, 20)  # Discretization 20 steps in 2 meters
    xs0 = [xxs[0] + step * math.cos(theta0) for step in steps]
    zs0 = [zzs[0] - step * math.sin(theta0) for step in steps]
    xs1 = [xxs[1] + step * math.cos(theta1) for step in steps]
    zs1 = [zzs[1] - step * math.sin(theta1) for step in steps]
    distances = [math.sqrt((xs0[idx] - xs1[idx]) ** 2 + (zs0[idx] - zs1[idx]) ** 2) for idx, _ in enumerate(xs0)]
    if np.min(distances) <= max(distances[0] / 1.5, min_distance):
        return True
    return False


def get_pifpaf_outputs(annotations):
    """Extract keypoints sets and scores from output dictionary"""
    if not annotations:
        return [], []
    keypoints_sets = np.array([dic['keypoints'] for dic in annotations]).reshape(-1, 17, 3)
    score_weights = np.ones((keypoints_sets.shape[0], 17))
    score_weights[:, 3] = 3.0
    # score_weights[:, 5:] = 0.1
    # score_weights[:, -2:] = 0.0  # ears are not annotated
    score_weights /= np.sum(score_weights[0, :])
    kps_scores = keypoints_sets[:, :, 2]
    ordered_kps_scores = np.sort(kps_scores, axis=1)[:, ::-1]
    scores = np.sum(score_weights * ordered_kps_scores, axis=1)
    return keypoints_sets, scores


@contextmanager
def bird_canvas(args, output_path):
    fig, ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    output_path = output_path + '.bird.png'
    x_max = args.z_max / 1.5
    ax.plot([0, x_max], [0, args.z_max], 'k--')
    ax.plot([0, -x_max], [0, args.z_max], 'k--')
    ax.set_ylim(0, args.z_max + 1)
    yield ax
    fig.savefig(output_path)
    plt.close(fig)
    print('Bird-eye-view image saved')