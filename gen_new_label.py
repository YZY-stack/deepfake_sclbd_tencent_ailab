import os
import cv2
import glob
from tqdm import tqdm
from itertools import combinations

MODE = 'train'
IMG_PATH = '/mntnfs/sec_data2/yanzhiyuan/HQ'
BASE_PATH = os.path.join(IMG_PATH, MODE)
SAVE_PATH = 'new_label_imgs'
os.makedirs(SAVE_PATH, exist_ok=True)

alpha = 0.5
visualize = False

fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
real_list = ['real']

all_list = fake_list + real_list

# for FF++, C52 = 10
combine = list(combinations(all_list, 2))
print(combine)

for comb in combine:
    print(comb)
    data1, data2 = comb
    combination_name = f'{data1}_{data2}'

    data1_video_list = glob.glob(
        os.path.join(BASE_PATH, 
            'fake' if data1 != 'real' else 'real', 
             data1 if data1 != 'real' else '', '*'
        )
    )
    data2_video_list = glob.glob(
        os.path.join(BASE_PATH, 
            'fake' if data2 != 'real' else 'real', 
             data2 if data2 != 'real' else '', '*'
        )
    )
    assert len(data1_video_list) == len(data2_video_list), "two video list should be the same."

    # 按顺序依次遍历1000个videos，并一一匹配
    for i in tqdm(range(len(data1_video_list))):
        video1 = data1_video_list[i]
        video2 = data2_video_list[i]
        # 拿到video name
        # 如果有real
        if video1.split('/')[-1] != video2.split('/')[-1]:
            # 取fake的video name instead of real的
            video_name = max(video1.split('/')[-1], video2.split('/')[-1])
        # 如果没有real，两个都一样
        else:
            video_name = video1.split('/')[-1]
        # 保存路径
        img_save_path = os.path.join(
            BASE_PATH, 'fake', 'newlabel', combination_name, video_name
        )
        os.makedirs(img_save_path, exist_ok=True)

        # 对这两个video里面的所有imgs一一相加
        img_path_list_1 = sorted(glob.glob(os.path.join(video1, '*')))
        img_path_list_2 = sorted(glob.glob(os.path.join(video2, '*')))
        img_length = min(len(img_path_list_1), len(img_path_list_2))
        # if len(img_path_list_1) == len(img_path_list_2):
        #     continue
        # else:
        #     print('a')
        for j in range(img_length):
            img1 = cv2.cvtColor(
                cv2.imread(img_path_list_1[j]), 
                cv2.COLOR_BGR2RGB
            )
            img2 = cv2.cvtColor(
                cv2.imread(img_path_list_2[j]), 
                cv2.COLOR_BGR2RGB
            )
            # 融合结果，依据：A*alpha + B*(1-alpha)
            img3 = alpha * img1 + (1-alpha) * img2

            # 保存结果
            cv2.imwrite(
                os.path.join(img_save_path,
                f'img3_{i}_{j}_{data1}_{data2}.png'), img3
            )

            # 如果想可视化，那么每个combination和video都只保存一张图
            if visualize and i % 10 ==0 and j == 0:
                cv2.imwrite(
                    os.path.join(SAVE_PATH, 
                    f'img1_{i}_{j}_{data1}_{data2}.png'), img1
                )
                cv2.imwrite(
                    os.path.join(SAVE_PATH, 
                    f'img2_{i}_{j}_{data1}_{data2}.png'), img2
                )
                cv2.imwrite(
                    os.path.join(SAVE_PATH, 
                    f'img3_{i}_{j}_{data1}_{data2}.png'), img3
                )
                break
        if visualize:
            break
