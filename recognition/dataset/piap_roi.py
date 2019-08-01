import cv2
import numpy as np
import os

def pchar_names(object_index):
    with open("./pchar84.names", "r") as f:
        names = f.readlines()

    object_name = names[object_index].split("\n")[0]

    return object_name

def main():
    img_path = "/home/jaehyeon/project/Drone/dev/PIAP/img/"
    img_list = [x for x in os.listdir(img_path) if x.split(".")[1] == "jpg"]
    total_image = len(img_list)
    total_cnt = 0
    for i in img_list:
        img_name = img_path + i
        txt_name = img_path + i.split(".")[0] + ".txt"

        img = cv2.imread(img_name,0)
        
        # Real value
        height = img.shape[0]
        width = img.shape[1]

        with open(txt_name, 'r') as f:
            cnt = 0
            while True:
                line = f.readline()
                if line == "": break

                split_line = line.split(' ')

                # Object Index
                obj_idx = split_line[0]

                # Object Position
                cx = float(split_line[1]) * width
                cy = float(split_line[2]) * height
                w = int(float(split_line[3]) * width)
                h = int(float(split_line[4]) * height)

                # centor to Top left, right
                x = int(cx - (w/2))
                y = int(cy - (h/2))

                # roi image
                roi_img = img[y:y+h, x:x+w]

                obj_name = pchar_names(int(obj_idx))

                save_path = "img/" + obj_name + "/" + i.split(".")[0] + "_" + str(cnt) + ".jpg"
                
                if os.path.exists("img/" + obj_name) is True:
                    if os.path.exists(save_path) is False:
                        cv2.imwrite(save_path, roi_img)

                else:
                    os.mkdir("img/" + obj_name)
                    if os.path.exists(save_path) is False:
                        cv2.imwrite(save_path, roi_img)

                cnt += 1

        total_cnt += 1
        if total_cnt % 1000 == 0:
            print(str(total_cnt) + "/" + str(total_image))

    print("=== Finished ===")

def count_roi_image():
    img_folder = "./img/"
    class_list = os.listdir(img_folder)

    num_image = 0
    for c_folder in class_list:
        img_list = os.listdir(img_folder + c_folder)
        num_image += len(img_list)

    print("total image : ", num_image)
        




if __name__ == '__main__':
    main()
