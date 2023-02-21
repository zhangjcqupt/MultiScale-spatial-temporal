# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time

import os
import torch.nn.parallel
import torch.optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ops.dataset_fusion import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
from thop import profile, clever_format

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import cv2
import cv2.cv2
from torch.autograd import Variable


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
parser.add_argument('--img_feature_dim',type=int, default=224)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--ope', default='baseline', type=str, help='operations for backbone(default:baseline)')
parser.add_argument('--VAP', default=False, action="store_true", help='use VAP for various-timescale aggregation')

#cam parameter
parser.add_argument('--use-cuda', action='store_true', default=False,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument('--aug_smooth', action='store_true', default=True,
                    help='Apply test time augmentation to smooth the CAM')
parser.add_argument(
    '--eigen_smooth',
    action='store_true',
    default=True,
    help='Reduce noise by taking the first principle componenet'
         'of cam_weights*activations')
parser.add_argument('--method', type=str, default='gradcam',
                    choices=['gradcam', 'gradcam++',
                             'scorecam', 'xgradcam',
                             'ablationcam', 'eigencam',
                             'eigengradcam', 'layercam', 'fullgrad'],
                    help='Can be gradcam/gradcam++/scorecam/xgradcam'
                         '/ablationcam/eigencam/eigengradcam/layercam')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]='3'

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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(path + "---Done---")
    else:
        print(path + "---This is the folder---")

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
        data_length = 1
    elif 'Lite' in this_weights:
        modality = 'Lite'
        data_length = 4
    elif 'PA' in this_weights:
        modality = 'PA'
        data_length = 4
    else:
        modality = 'Flow'
        data_length = 5
    this_arch = this_weights.split('TSN_')[1].split('_')[2]
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

    input = torch.randn(8, 3, 224, 224)
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:{}, Params:{}'.format(flops, params))

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
                       new_length=data_length, #1 if modality == "RGB" else 5,
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

def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():

        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality in ['PA', 'Lite']:
            length = 12
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality " + modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

real_test_npy = np.zeros(shape=(total_num, num_class))
sftp_ppkeke = torch.nn.Softmax(dim=1)
gt_npy = np.zeros(shape=total_num, dtype=np.int)

top1 = AverageMeter()
top5 = AverageMeter()


for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    for _ in range(1):
    # with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        this_label = None

        for n_seg, (_, (index, (data, label, indices, video_path))), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            target_layers = [net.module.base_model.layer4[-1]]
            out_image_rootpath = '/data/zhangj/Dataset/torch_cam/TSN/'

            data = Variable(data, requires_grad=True)

            image_rgb_list = Img_rgb(indices, video_path)
            target_category = None  # cam start
            methods = \
                {"gradcam": GradCAM, "scorecam": ScoreCAM, "gradcam++": GradCAMPlusPlus,
                 "ablationcam": AblationCAM, "xgradcam": XGradCAM, "eigencam": EigenCAM,
                 "eigengradcam": EigenGradCAM, "layercam": LayerCAM, "fullgrad": FullGrad}
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=net, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
                # grayscale_cam = cam(input_tensor=data, target_category=target_category, aug_smooth=args.aug_smooth,
                #                     eigen_smooth=args.eigen_smooth)
                grayscale_cam_no = cam(input_tensor=data, target_category=target_category, aug_smooth=False,
                                       eigen_smooth=False)
                for i in range(8):
                    # grayscale_i = grayscale_cam[i, ::]
                    grayscale_i_no = grayscale_cam_no[i, ::]
                    path = image_rgb_list[i]
                    a = path.split('/')
                    out_image_path = out_image_rootpath + a[-4] + '/' + a[-3] + '/' + a[-2]
                    mkdir(out_image_path)
                    img_rgb = cv2.resize(cv2.imread(path, 1)[:, :, ::-1], (224, 224))
                    img_rgb = np.float32(img_rgb) / 255  # oringinal is (0~255),now is (0~1)
                    # cam_image = show_cam_on_image(img_rgb, grayscale_i, use_rgb=True)
                    # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    cam_image_no = show_cam_on_image(img_rgb, grayscale_i_no, use_rgb=True)
                    cam_image_no = cv2.cvtColor(cam_image_no, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(out_image_path + '/' + '{}'.format(a[-1]), cv2.cv2.imread(path))
                    # cv2.imwrite(out_image_path + '/' + '{}_{}_smooth'.format(a[-1], args.method), cam_image)
                    cv2.imwrite(out_image_path + '/' + '{}_{}'.format(args.method, a[-1] ), cam_image_no)

            rst = eval_video((i, data, label), net, n_seg, modality)
            this_rst_list.append(rst[1])
            this_label = label
            # new_fc_weight = net.state_dict()['new_fc.weight'].cpu().numpy()
        assert len(this_rst_list) == len(coeff_list)

        for i_coeff in range(len(this_rst_list)):
            this_rst_list[i_coeff] *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))


        index = index.numpy()
        real_test_npy[index, :]=sftp_ppkeke(torch.from_numpy(ensembled_predict)).numpy()[:, :]
        gt_npy[index] = this_label.numpy()

        top1.update(prec1.item(), this_label.numel())
        top5.update(prec5.item(), this_label.numel())
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))

video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]

# np.save('test_flow.npy', real_test_npy)
# np.save('gt_flow.npy', gt_npy)

if args.csv_file is not None:
    print('=> Writing result to csv file: {}'.format(args.csv_file))
    with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
        categories = f.readlines()
    categories = [f.strip() for f in categories]
    with open(test_file_list[0]) as f:
        vid_names = f.readlines()
    vid_names = [n.split(' ')[0] for n in vid_names]
    assert len(vid_names) == len(video_pred)
    if args.dataset != 'somethingv2':  # only output top1
        with open(args.csv_file, 'w') as f:
            for n, pred in zip(vid_names, video_pred):
                f.write('{};{}\n'.format(n, categories[pred]))
    else:
        with open(args.csv_file, 'w') as f:
            for n, pred5 in zip(vid_names, video_pred_top5):
                fill = [n]
                for p in list(pred5):
                    fill.append(p)
                f.write('{};{};{};{};{};{}\n'.format(*fill))

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
    plt.xticks(xlocations, labels, rotation=-45)
    plt.tick_params(axis='x', labelsize=7)
    plt.yticks(xlocations, labels)
    plt.tick_params(axis='y', labelsize=7)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('Predicted label')
    # plt.xlabel('True label')

cf = confusion_matrix(video_labels, video_pred).astype(float)
# cf = confusion_matrix(video_pred, video_labels).astype(float)
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

plot_confusion_matrix(cf_normalized, title='confusion matrix')
save_path = os.path.join(root_path, 'confusion_matrix')
if not os.path.exists(save_path):
    os.mkdir(save_path)
# plt.savefig(os.path.join(save_path, 'TSN_1frame_50epoch_midframecrop(sec).png'), format='png')
#plt.show()

#np.save('cm.npy', cf)
cls_cnt = cf.sum(axis=1)
cls_pre = cf.sum(axis=0)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
print(cls_acc * 100)
# cls_pred = cls_hit / cls_pre
# print(cls_pred)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))


