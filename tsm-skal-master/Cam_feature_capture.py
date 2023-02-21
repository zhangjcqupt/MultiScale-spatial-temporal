# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing
import matplotlib.pyplot as plt
import argparse
import time
import os
import cv2

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

parser.add_argument('--VAP', default=False, action="store_true", help='use VAP for various-timescale aggregation')
parser.add_argument('--energy_thr', default=0.7, type=float)
parser.add_argument('--mode', default='s1', type=str, choices=['s1', 's2'])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]='0'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def label2cls(root):
    f = open(os.path.join(root, 'categories.txt'), 'r')
    line = f.readline()
    label2cls_list ={}
    cnt = 0
    while line:
        line = line.strip('\n')
        # cls, label = line.split(' ')
        label2cls_list[str(cnt)] = str(line)
        cnt += 1
        line = f.readline()
    return label2cls_list

def Img_rgb(idex, video_path):
    images = []
    dirs = os.listdir(video_path[0])
    dirs.sort()
    for p in idex:
        for i in p:
            q = int(i)
            images.append(video_path[0] + '/' + dirs[q-1])

    return images


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSM_')[1].split('_')[2]
    modality_list.append(modality)
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class,  this_test_segments, modality,
              mode=args.mode,
              energy_thr=args.energy_thr,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              has_vap=args.VAP
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    input_size = net.scale_size if args.full_res else net.input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=1 if modality == "RGB" else 5,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
    )

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)

    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)

    data_iter_list.append(data_gen)
    net_list.append(net)


output = []

def save_image(image_s1, heat_map, label, img_save_path):
    img_h, img_w = image_s1.shape[:2]
    heat_map = heat_map.reshape(heat_map.shape[1], heat_map.shape[2], 1)

    # feat_map = (heat_map - 0.5) + 1
    # feat_map = np.uint8(np.float32(feat_map))
    # feat_map = cv2.resize(feat_map, (img_w, img_h))
    # feat_map_featrue = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
    # feat_map_featrue = np.float32(feat_map_featrue)

    heat_map = cv2.resize(heat_map, (img_w, img_h))
    heat_map = cv2.applyColorMap(np.uint8(255 * heat_map), cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map)

    cv2.imwrite(img_save_path, np.uint8(image_s1))


    feat_map_featrue = image_s1*0.0 + heat_map*1.0
    cv2.imwrite(img_save_path.replace('.tif', '_heatmap.tif').replace('.jpg', '_heatmap.jpg'),
                np.uint8(feat_map_featrue))

    feat_map_featrue = image_s1 * 0.5 + heat_map * 0.5
    cv2.imwrite(img_save_path.replace('.tif', '_heatmap.tif').replace('.jpg', '_heatmap_color.jpg'),
                np.uint8(feat_map_featrue))

    feature_fusion = (1.5 * heat_map[:, :, 2] + 0.15 * heat_map[:,:,1] + 0.05 * heat_map[:,:,0]) / 255.0
    feature_fusion_1 = np.zeros(shape=feature_fusion.shape, dtype=np.float32)
    feature_fusion = np.maximum(feature_fusion, feature_fusion_1)
    feature_fusion_1[:, :] = 1.0
    feature_fusion = np.minimum(feature_fusion, feature_fusion_1)
    feature_fusion = np.repeat(feature_fusion, 3, axis=1).reshape(image_s1.shape)
    feature_fusion_3 = image_s1 * feature_fusion

    cv2.imwrite(img_save_path.replace('.tif', '_heatmap.tif').replace('.jpg', '_heatmap_gray.jpg'),
                np.uint8(feature_fusion_3))



def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        label2cls_list = label2cls(root_path)
        i, image, label, indices, record = video_data
        batch_size = label.numel()

        img_save_dir = './attvisual_image'
        img_true_save_dir = './attvisual_image/' + args.dataset + '/true'
        img_false_save_dir = './attvisual_image/' + args.dataset + '/false'
        sta_save_dir = './save_status'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        if not os.path.exists(img_true_save_dir):
            os.makedirs(img_true_save_dir)
        if not os.path.exists(img_false_save_dir):
            os.makedirs(img_false_save_dir)
        if not os.path.exists(sta_save_dir):
            os.makedirs(sta_save_dir)

        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        image_in = image.view(-1, length, image.size(2), image.size(3))
        if is_shift:
            image_in = image_in.view(batch_size * num_crop, this_test_segments, length, image.size(2), image.size(3))
        rst, heat_map = net(image_in, is_training=True)

        heat_map = heat_map.cpu().detach().numpy()
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        cnt = 0

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        _, pred = torch.max(rst, dim=1)
        pred = int(pred.cpu().detach().numpy())
        rst = rst.data.cpu().numpy().copy()

        # for i in range(batch_size):
        for i in range(this_test_segments):
            image_rgb_list = Img_rgb(indices, record)
            path = image_rgb_list[i]
            img = Image.open(path).convert('RGB')

            cnt += 1

            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            if pred != int(label):
                image_file = img_save_dir + '/' + path.split('/')[-3] + '/' + path.split('/')[-2]
                image_file = image_file.replace('attvisual_image/', 'attvisual_image/' + args.dataset +'/false/')
                if not os.path.exists(image_file):
                    os.makedirs(image_file)
                img_save_path = image_file + '/' + path.split('/')[-1]
                img_save_path = img_save_path.replace('.jpg', '_wrong_{}.jpg'.format(label2cls_list[str(pred)]))
                save_image(img, heat_map[i], label, img_save_path)
            else:
                image_file = img_save_dir + '/' + path.split('/')[-3] + '/' + path.split('/')[-2]
                image_file = image_file.replace('attvisual_image/', 'attvisual_image/' + args.dataset + '/true/')
                if not os.path.exists(image_file):
                    os.makedirs(image_file)
                img_save_path = image_file + '/' + path.split('/')[-1]
                save_image(img, heat_map[i], label, img_save_path)



        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()

for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        this_label = None
        for n_seg, (_, (image, label, indices, video_path)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            record = []
            for image_path in video_path:
                load_image_path = os.path.join(root_path, image_path)
                record.append(load_image_path)
            _, rst, label = eval_video((i, image, label, indices, record), net, n_seg, modality)
            # this_rst_list.append(rst[1])
            this_rst_list.append(rst)
            this_label = label

        assert len(this_rst_list) == len(coeff_list)
        for i_coeff in range(len(this_rst_list)):
            this_rst_list[i_coeff] *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
        top1.update(prec1.item(), this_label.numel())
        top5.update(prec5.item(), this_label.numel())
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))

video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]


labels = []
file = open(os.path.join(root_path, 'categories.txt'), 'r')
lines = file.readlines()
for line in lines:
    labels.append(line.strip())
file.close()

tick_marks = np.array(range(len(labels))) + 0.5
def plot_confusion_matrix(cm, title='Confusion', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=-90)
    plt.tick_params(axis='x', labelsize=7)
    plt.yticks(xlocations, labels)
    plt.tick_params(axis='y', labelsize=7)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('Predicted label')
    # plt.xlabel('True label')

cf = confusion_matrix(video_labels, video_pred).astype(float)

np.set_printoptions(precision=2)
cf_normalized = cf / cf.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cf_normalized[y_val][x_val]
    if c >= 0.5:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='w', fontsize=7, va='center', ha='center')
    elif c >= 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=7, va='center', ha='center')

plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cf_normalized, title='')
save_path = os.path.join(root_path, 'confusion_matrix')
if not os.path.exists(save_path):
    os.mkdir(save_path)
plt.savefig(os.path.join(save_path, 'TSN_local_VAP_s2_RGB_16_epoch40_mod20.png'), format='png')

np.save('cm.npy', cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

kappa_value = cohen_kappa_score(video_labels, video_pred)
print('kappa %f' % kappa_value)
