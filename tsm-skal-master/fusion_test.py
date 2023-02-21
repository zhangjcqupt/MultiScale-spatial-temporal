import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json


def dfs(start, n, lens, data, gt, weight, labels, gt_root, p=[]):
    if lens <= 0:
        q = np.zeros(shape=data[0].shape, dtype=np.float32)
        strs = ""
        for i in p:
            if len(weight) > 0:
                q = q + data[i] * weight[i]
            else:
                q = q + data[i]
            strs += '{}'.format(i+1)
        q = torch.from_numpy(q)
        q = torch.max(q, 1)[1]

        # tick_marks = np.array(range(len(labels))) + 0.5
        # def plot_confusion_matrix(cm, title='Confusion', cmap=plt.cm.binary):
        #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #     plt.title(title)
        #     plt.colorbar()
        #     xlocations = np.array(range(len(labels)))
        #     plt.xticks(xlocations, labels, rotation=-60)
        #     plt.tick_params(axis='x', labelsize=7)
        #     plt.yticks(xlocations, labels)
        #     plt.tick_params(axis='y', labelsize=7)
        #     plt.ylabel('Predicted label')
        #     plt.xlabel('True label')
        #
        # cf = confusion_matrix(q, gt).astype(float)
        # np.set_printoptions(precision=2)
        # cf_normalized = cf / cf.sum(axis=1)[:, np.newaxis]
        # plt.figure(figsize=(12, 8), dpi=120)
        # ind_array = np.arange(len(labels))
        # x, y = np.meshgrid(ind_array, ind_array)
        #
        # for x_val, y_val in zip(x.flatten(), y.flatten()):
        #     c = cf_normalized[y_val][x_val]
        #     if c >= 0.5:
        #         plt.text(x_val, y_val, "%0.2f" % (c,), color='w', fontsize=7, va='center', ha='center')
        #     elif c >= 0.01:
        #         plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')
        #
        # plt.gca().set_xticks(tick_marks, minor=True)
        # plt.gca().set_yticks(tick_marks, minor=True)
        # plt.gca().xaxis.set_ticks_position('none')
        # plt.gca().yaxis.set_ticks_position('none')
        # plt.grid(True, which='minor', linestyle='-')
        # plt.gcf().subplots_adjust(bottom=0.15)
        #
        # plot_confusion_matrix(cf_normalized, title='TSN_fusion confusion matrix')
        # root = gt_root.split('.')[0]
        # save_path = root + '_confusion'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # plt.savefig(os.path.join(save_path, 'TSN_fusion'+'_'+strs+'.png'), format='png')

        running_corrects = torch.sum(q.view(-1) == gt).tolist() / gt.shape[0]

        print("{}: {}".format(strs, running_corrects))
        return
    for i in range(start+1, n):
        dfs(i, n, lens - 1, data, gt, weight, labels, gt_root, p + [i])


def load_result(root, gt_root, category,weight=[]):
    data = []
    labels = []
    gt = torch.from_numpy(np.load(gt_root))
    file = open(category, 'r')
    lines = file.readlines()
    for line in lines:
        labels.append(line.strip())
    file.close()
    print("len root = {}".format(len(root)))
    for file_name in root:
        s = np.load(file_name)
        n = s.shape[0]
        for i in range(n):
            t = 0.99999 + np.sum(s[i])
            s[i] /= t
        data.append(s)
    if len(weight) != len(data):
        weight = []
    for lens in range(1, 1+len(data)):
        print("lens = {}".format(lens))
        dfs(-1, len(data), lens, data, gt, weight, labels, gt_root)

def rebuild_test_npy(npy_dir, src_txt_dir, dst_txt_dir):
    cp = {}
    with open(src_txt_dir, "r") as fr:
        lines = fr.readlines()
        idx = 0
        for line in lines:
            v_name = line.split(' ')[0].split('/')[-1]
            cp[v_name] = idx
            idx += 1
    id_map_new_id = [i for i in range(len(cp))]
    with open(dst_txt_dir, "r") as fr:
        lines = fr.readlines()
        idx = 0
        for line in lines:
            v_name = line.split(' ')[0].split('/')[-1]
            # id_map_new_id[idx] = cp[v_name]
            id_map_new_id[cp[v_name]] = idx
            idx += 1
    npy_data = np.load(npy_dir)
    new_data = np.zeros(shape=npy_data.shape)
    new_data[:, :] = npy_data[id_map_new_id, :]
    np.save(npy_dir.split('.npy')[0] + '_new.npy', new_data)

if __name__ == '__main__':

    # rebuild_test_npy(r"/data_ssd/huangsixiang/HMDB51/TSM_model/FLOW/TSM_hmdb51_Flow_resnet50_shift8_blockres_avg_segment8_e80_None/test.npy",
    #                  r"/data_ssd/huangsixiang/HMDB51/TSM/rgt_rgb/test.txt",
    #                  r"/data_ssd/huangsixiang/HMDB51/TSM/rgt_flow/test.txt")



    load_result([r"/raid/zhangj/Code/TSM/tsm_SKAL/tsm-skal-master/scripts/checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s1_67/test_RGB_PYKAL_att_VAP_s1_16.npy",
                 r"/raid/zhangj/Code/TSM/tsm_SKAL/tsm-skal-master/checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s2_59/test_RGB_PYKAL-VAP_s2_16.npy",
                 r"/raid/zhangj/Code/TSM/tsm_SKAL/tsm-skal-master/scripts/checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s2/test_RGB_PYKAL_local_VAP_s2_16.npy"
                 ],
                 r"/raid/zhangj/Code/TSM/tsm_SKAL/tsm-skal-master/scripts/checkpoint/TSM_era_RGB_pyconv_avg_segment16_e40_VAP_s1_67/gt_RGB_PYKAL_att_VAP_s1_16.npy",
                r"/raid/zhangj/Dataset/ERA_Dataset/Images/categories.txt",
                [1, 0, 1])
    print("done")

#/raid/zhangj/Code/TSM/temporal-shift-module-master/scripts/checkpoint/TSM_era_Flow_resnet50_avg_segment16_e50/test_flow_16.npy
# TSM_era_RGB_resnet50_avg_segment8_e25_vap
# TSM_era_RGB_resnet50_shift8_blockres_avg_segment16_e25