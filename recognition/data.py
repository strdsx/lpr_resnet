import cv2
import os

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ga", "na",
    "go", "no", "da", "la", "ma", "geo", "neo", "deo",
    "leo", "meo", "do", "lo", "mo", "goo", "noo", "doo", "loo", "moo",
    "beo", "seo", "uh", "jeo", "bo", "so", "oh", "jo", "boo", "soo", "woo",
    "joo", "heo", "ba", "sa", "ah", "ja", "bae", "ha", "ho", "wool",
    "san", "dae", "in", "cheon", "gwang", "jeon", "gyeong", "gee",
    "gang", "won", "choong", "book", "nam", "je", "se", "jong"]

def data_ResNet_to_YOLO():
    data_path = 'data/rename_gray_headline_piap/'
    classes = os.listdir(data_path)
    for c in classes:
        # rename_gray_headline_piap/choong/
        class_path = data_path + c + '/'
        imagelist = os.listdir(class_path)
        for i in imagelist:
            image_full_path = class_path + i
            img = cv2.imread(image_full_path)
            if type(img) == type(None):
                os.remove(image_full_path)
                print("Removed ==>", image_full_path)
            else:
                save_path = "data/img/" + c + "_" + i # data/obj/choong_piap_77.jpg

                name = i.split('.')[0] # piap_77
                name_txt = c + "_" + name + ".txt" # choong_piap_77.txt
                obj_number = class_names.index(c)
                yolo_roi = "0.499999 0.499999 0.999999 0.999999"

                cv2.imwrite(save_path, img)
                with open("data/img/"+name_txt, 'w') as f:
                    line = str(obj_number) + " " + yolo_roi
                    f.write(line)

def YOLO_train_txt():
    imagelist = list(x for x in os.listdir("./data/img/") if x.split('.')[1] == "jpg")
    with open("data/trainplatechar_headline.txt", 'w') as f:
        for i in imagelist:
            line = "data/img/" + i
            f.write(line)

if __name__ == '__main__':
	data_ResNet_to_YOLO()
	YOLO_train_txt()
