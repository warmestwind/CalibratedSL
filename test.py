import torch
from config import get_config
import os
import glob
import cv2
from utils.imutils import img_pad, invert_pad
import shutil
from utils.plots import *

import pandas as pd

import json


config = get_config()
config.device = 'cuda'
config.is_train = False

num_lands = 4


def remap_p(p, shape, pad_h, pad_w, new_h, new_w):
    r_h = new_h / shape[0]
    r_w = new_w / shape[1]
    p[0] = int(p[0] * r_h + pad_h)
    p[1] = int(p[1] * r_w + pad_w)
    p = np.append(p, 1)
    return p



def get_json_point(js_path):
    with open(js_path, 'r', encoding='utf-8') as f:
        j_dict = json.load(f)

    for p in j_dict['shapes']:
        if p['label'] == '1':
            pt = np.round(p['points'][0][::-1])
        if p['label'] == '2':
            pb = np.round(p['points'][0][::-1])
        if p['label'] == '3':
            p1 = np.round(p['points'][0][::-1])  # h, w
        if p['label'] == '4':
            p2 = np.round(p['points'][0][::-1])  # h, w
        if p['label'] == '5':
            p3 = np.round(p['points'][0][::-1])  # h, w
        if p['label'] == '6':
            p4 = np.round(p['points'][0][::-1])  # h, w
    return pt, pb, p1, p2, p3, p4




def main(ckpt_path, draw=False):

    with torch.no_grad():
        ckpt_epoch = ckpt_path.split('\\')[-1].split('.')[0].split('_')[-1]
        model = Unet(config)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model.cuda()
        model.eval()

        if os.path.exists('./predictions__plax'):
            shutil.rmtree('./predictions__plax')
        os.mkdir('./predictions_plax')

        data_roots = [r'D:\Dataset\plax_img_kf_6']


        for data_root in data_roots:
            df = pd.read_csv(r'D:\Dataset\spa_plax1.csv')

            fold = 0
            r_train = 0.8
            r_val = 0.1

            tmps = glob.glob(data_root+'\\*.jpg')
            tmps = sorted(tmps)

            start = int(fold * len(tmps) / 5)
            tmps = tmps[start:] + tmps[:start]

            train_ids = tmps[:int(r_train * len(tmps)) + 1]
            val_ids = tmps[int(r_train * len(tmps)) + 1: int(r_train * len(tmps)) + 1 + int(r_val * len(tmps)) + 1]
            test_ids = tmps[int(r_train * len(tmps)) + 1 + int(r_val * len(tmps)) + 1:]

            ################################
            pids = val_ids #  test_ids

            for pid in pids:

                id = pid.split('\\')[-1]
                print(id)

                idood_cls = {}


                imgs_ori = cv2.imread(pid, 0)
                ori_shape = imgs_ori.shape
                img, pad_h, pad_w, new_h, new_w, _ = img_pad(imgs_ori)
                imgs = img / 255
                js_files = pid.replace('.jpg', '.json')
                _, _, l, r, b, c = get_json_point(js_files)



                batch_input = torch.from_numpy(imgs[np.newaxis,  np.newaxis, ...]).float().cuda()

                results = model.forward(batch_input)
                global_results = results['alpha'] / results['alpha'].sum(dim= 1, keepdim =True)


                img3 = cv2.cvtColor(imgs_ori, cv2.COLOR_GRAY2BGR)

                # draw tmp local
                spa = df[df['PID'].str.contains(id.split('_')[0])]['spacing'].to_list()[0]


                out_mean = global_results.cpu().numpy()

                pred_coords = []
                tmp_max = []
                for i in range(num_lands):

                    heatmap_i = out_mean[:, i]
                    idood_cls['maxp_p'+str(i)] = heatmap_i.max()
                    tmp_max.append(heatmap_i.max())
                    x, y = np.unravel_index(np.argmax(heatmap_i), out_mean.shape)[-2:]

                    pred_coords.append((x, y))

                pred_coords = np.stack(pred_coords)
                pred_coords = invert_pad(pred_coords, pad_h, pad_w, ori_shape).astype(np.int32)

                if draw:
                    # visualization
                    img3[int(l[0]) - 3:int(l[0]) + 3, int(l[1]) - 3:int(l[1]) + 3] = np.array([0, 255, 0])
                    img3[int(r[0]) - 3:int(r[0]) + 3, int(r[1]) - 3:int(r[1]) + 3] = np.array([0, 255, 0])
                    img3[int(b[0]) - 3:int(b[0]) + 3, int(b[1]) - 3:int(b[1]) + 3] = np.array([0, 255, 0])
                    img3[int(c[0]) - 3:int(c[0]) + 3, int(c[1]) - 3:int(c[1]) + 3] = np.array([0, 255, 0])

                    img3[pred_coords[0][0] - 3:pred_coords[0][0] + 3,
                    pred_coords[0][1] - 3:pred_coords[0][1] + 3] = np.array([0, 255, 255])  # yellow
                    img3[pred_coords[1][0] - 3:pred_coords[1][0] + 3,
                    pred_coords[1][1] - 3:pred_coords[1][1] + 3] = np.array([0, 255, 255])
                    img3[pred_coords[2][0] - 3:pred_coords[2][0] + 3,
                    pred_coords[2][1] - 3:pred_coords[2][1] + 3] = np.array([0, 255, 255])
                    img3[pred_coords[3][0] - 3:pred_coords[3][0] + 3,
                    pred_coords[3][1] - 3:pred_coords[3][1] + 3] = np.array([0, 255, 255])


                    cv2.imwrite('./predictions_plax' + '/' + id + '_ori_gt' + '.jpg', img3)


            return  ckpt_epoch


if __name__ == '__main__':

    from model.U_CSL_US import Unet
    root_dir = r'D:\Code\github\uncertainty\ckpt\us'

    ckpts_dirs = glob.glob(root_dir+'\\*.pt')
   
    for ckpt in ckpts_dirs:
        print(ckpt)
        temp_epoch = main(ckpt, True)
       
