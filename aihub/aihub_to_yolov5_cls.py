import os
import argparse
import json
import numpy as np
import skimage.io


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', help="image directory", required=True)
    parser.add_argument('-l', '--label_dir', help="label directory", required=True)
    parser.add_argument('-o', '--output_dir', help="output directory", required=True)
    parser.add_argument('-t', '--type', choices=['train', 'val'], help='select train or val', required=True)
    _args = parser.parse_args()
    return _args


def get_files_path(path, d):
    if os.path.isfile(path) and os.path.splitext(path)[-1] in ['.jpg', '.jpeg', '.png', '.json']:
        f_name = os.path.splitext(os.path.basename(path))[0]
        d[f_name] = path
        return
    dirs = os.listdir(path)
    for _dir in dirs:
        get_files_path(os.path.join(path, _dir), d)


def main():
    args = arg_parse()
    input_dir = args.image_dir
    label_dir = args.label_dir
    data_type = args.type
    output_dir = os.path.join(args.output_dir, data_type)
    o_image_dir = os.path.join(output_dir, 'images')
    o_label_dir = os.path.join(output_dir, 'labels')

    #if not os.path.exists(output_dir):
    #    os.makedirs(o_image_dir)
    #    os.makedirs(o_label_dir)


    images = {}
    get_files_path(input_dir, images)
    #[print(i, images[i]) for i in images]
    labels = {}
    get_files_path(label_dir, labels)
    #[print(l, labels[l]) for l in labels]

    # 저장경로 생성
    for k, v in NEED_CLASS.items():
        if not os.path.exists(os.path.join(output_dir, v)):
            os.makedirs(os.path.join(output_dir, v))

    num_images = 0


    for idx, (label, label_path) in enumerate(labels.items()):
        # 해당 라벨에 해당하는 이미지가 존재하는지 체크
        if label not in images:
            continue


        try:
            # 라벨 파싱
            with open(label_path, encoding='utf8') as f:
                j_label = json.load(f)
            _img_width = j_label["image"]["resolution"][0]
            _img_height = j_label["image"]["resolution"][1]
            check_images = False
            for obj in j_label["annotations"]:
                # 박스없는건 넘어가기
                if 'box' not in obj:
                    continue
                if not check_images:
                    num_images += 1
                    check_images = True

                _img_class_orig = obj["class"]
                _img_coord = obj["box"]

                # 원하는 클래스 인지 체크
                if _img_class_orig not in NEED_CLASS:
                    continue
                # 비정상적 사이즈는 아닌지 체크
                if _img_coord[3]-_img_coord[1] == 0 or _img_coord[2]-_img_coord[0] == 0:
                    continue


                # 이미지 크롭하여 생성
                # print(images[label])
                img_arr = skimage.io.imread(images[label], as_gray=False)
                # print(img_arr.shape, type(img_arr), _img_coord, _img_class_orig)
                crop_img = img_arr[_img_coord[1]:_img_coord[3], _img_coord[0]:_img_coord[2]]
                # print(crop_img.shape, type(crop_img), np.max(crop_img), np.min(crop_img))

                # import cv2
                # cv2.imshow("-", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)

                # 크롭이미지 저장
                save_path = os.path.join(output_dir, NEED_CLASS[_img_class_orig], os.path.basename(images[label]))
                skimage.io.imsave(save_path, crop_img)
                num_classes[_img_class_orig] += 1
        except Exception as e:
            print(e)
            print(images[label])

        if idx % 100 == 0:
            print(f"{idx} image copied.")

    # 결과 파일 생성
    with open(os.path.join(output_dir, data_type+'_result.txt'), mode='w', encoding='utf8') as f:
        f.write(f"num_images: {num_images}\n")
        f.write(f"result: {num_classes}")
    print(f"num_images: {num_images} / result: {num_classes}")


if __name__ == '__main__':
    # AIHUB 고소작업자 안전 영상 데이터
    num_classes = {
        "05": 0,  # 안전화 착용
        "06": 0,  # 안전화 미착용
        #"07": 0,  # 안전모 착용
        #"08": 0   # 안전모 미착용
    }
    NEED_CLASS = {
        "05": "safety-shoes",     # 안전화 착용
        "06": "normal-shoes",     # 안전화 미착용
        #"07": "helmet",           # 안전모 착용
        #"08": "no-helmet"         # 안전모 미착용
    }
    main()
