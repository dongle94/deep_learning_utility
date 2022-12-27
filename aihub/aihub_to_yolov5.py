import os
import argparse
import json
import shutil


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', help="image directory", required=True)
    parser.add_argument('-l', '--label_dir', help="label directory", required=True)
    parser.add_argument('-o', '--output_dir', help="output directory", required=True)
    parser.add_argument('-t', '--type', choices=['train', 'val'], help='select train or val', required=True)
    _args = parser.parse_args()
    return _args


def get_files_path(path, d):
    if os.path.isfile(path) and os.path.splitext(path)[-1] in ['.jpg', '.json']:
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

    if not os.path.exists(output_dir):
        os.makedirs(o_image_dir)
        os.makedirs(o_label_dir)


    images = {}
    get_files_path(input_dir, images)
    #[print(i, images[i]) for i in images]
    labels = {}
    get_files_path(label_dir, labels)
    #[print(l, labels[l]) for l in labels]
    _classes = {
        "WO-01": 0,  # 작업자
        "WO-02": 0,  # 수신원
        "WO-04": 0,  # 안전모 미착용
        "WO-07": 0  # 안전화 미착용
    }
    for idx, (label, label_path) in enumerate(labels.items()):
        # 해당 라벨의 이미지가 존재하는지 체크
        if label not in images:
            continue

        # 라벨 파싱
        with open(label_path, encoding='utf8') as f:
            j_label = json.load(f)
        _img_width = j_label["Raw Data Info."]["resolution"][0]
        _img_height = j_label["Raw Data Info."]["resolution"][1]

        for obj in j_label["Learning Data Info."]["annotation"]:
            # 박스없는건 넘어가기
            if 'box' not in obj:
                continue
            _img_class_orig = obj["class_id"]
            _img_coord = obj["box"]
            rel_coord = [
                _img_coord[0] / _img_width,
                _img_coord[1] / _img_height,
                _img_coord[2] / _img_width,
                _img_coord[3] / _img_height
            ]

            # 원하는 클래스 인지 체크
            if _img_class_orig not in NEED_CLASS:
                continue

            # 이미지 복사
            shutil.copy(images[label], o_image_dir)

            # 라벨 생성
            f_name = os.path.join(o_label_dir, label + '.txt')
            with open(f_name, mode='a', encoding='utf8') as f:
                _classes[_img_class_orig] += 1
                txt = f'{NEED_CLASS[_img_class_orig]} {rel_coord[0]} {rel_coord[1]} {rel_coord[2]} {rel_coord[3]}\n'
                f.write(txt)

        if idx % 100 == 0:
            print(f"{idx} image copied.")

    # 결과 파일 생성
    with open(os.path.join(output_dir, data_type+'_result.txt'), mode='w', encoding='utf8') as f:
        f.write(f"result: {_classes}")
    print(f"result: {_classes}")


if __name__ == '__main__':
    # AIHUB 고소작업자 안전 영상 데이터
    NEED_CLASS = {
        "WO-01": 1,     # 작업자
        "WO-02": 1,     # 수신원
        "WO-04": 0,     # 안전모 미착용
        "WO-07": 2      # 안전화 미착용
    }
    main()
