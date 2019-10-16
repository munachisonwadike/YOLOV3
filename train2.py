import argparse
import test 
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models import *
from utils.datasets import *
from utils.utils import *
from visdom import Visdom

weights_dir = 'weights' + os.sep #weights directory
best = weights_dir + 'best.pt' # checkpoint of weights with best mAP
last = weights_dir + 'last.pt' # checkpoint of weights from most recent forward pass 
results_fl = 'results.txt'

# Hyperparameters 
hyp = { 'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
        'cls_pw': 1.446,  # cls BCELoss positive_weight
        'degrees': 1.113,  # image rotation (+/- deg)
        'fl_gamma': 0.5,  # focal loss gamma
        'giou': 1.582,  # giou loss gain
        'hsv_h': 0.01,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
        'iou_t': 0.2635,  # iou training threshold
        'lr0': 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
        'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
        'momentum': 0.97,  # SGD momentum
        'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
        'obj_pw': 3.941,  # obj BCELoss positive_weight
        'scale': 0.1059,  # image scale (+/- gain)
        'shear': 0.5768,  # image shear (+/- deg)
        'translate': 0.06797,  # image translation (+/- fraction)
        'weight_decay': 0.0004569 }  # optimizer weight decay
       
# Visdom class to visualise training loss
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        y=y.item()
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
plotter = VisdomLinePlotter(env_name='Plots')


# Prebias function to train output bias layers for 1 epoch and create new backbone
def run_prebias():
    if args.prebias:
        train()  # results saved to last.pt after 1 epoch
        backbone_generate(last)  # backbone is saved as backbone.pt (see utils/utils)
        args.weights = weights_dir + 'backbone.pt' # set train to continue from backbone.pt
        args.prebias = False  


# Main training function
def train():
    acc = args.accumulate  
    batch_size = args.batch_size
    cfg_file = args.cfg
    data = args.data
    if args.prebias:
        epochs = 1
    else:
        epochs = args.epochs 
       
    img_size = args.img_size
    weights = args.weights  

    if 'pw' not in args.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    # Initialize
    init_seeds()
    multi_scale = args.multi_scale

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_fl):
        os.remove(f)

    # Initialize model
    model = YOLOV3(cfg_file, arc=args.arc).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    if args.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.
    ####i edited the below lines
    try_download(weights) # will only run if not os.path.isfile(weights)
    
    if weights.endswith('.pt'):  # pytorch format
        # e.g are 'last.pt', 'yolov3-spp.pt'
    ####
        if args.bucket:
            os.system('gsutil cp gs://%s/last.pt %s' % (args.bucket, last))  # download from bucket
        chkpt = torch.load(weights, map_location=device)

        # load model
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_fl, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt


    elif len(weights) > 0:  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)

    if args.transfer or args.prebias:  # transfer learning edge (yolo) layers
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

        if args.prebias:
            for p in optimizer.param_groups:
                # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                p['lr'] *= 100  # lr gain
                if p.get('momentum') is not None:  # for SGD but not Adam
                    p['momentum'] *= 0.9

        for p in model.parameters():
            if args.prebias and p.numel() == nf:  # train (yolo biases)
                p.requires_grad = True
            elif args.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                p.requires_grad = True
            else:  # freeze layer
                p.requires_grad = False
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=args.rect,  # rectangular training
                                  image_weights=args.img_weights,
                                  cache_labels=False if epochs > 10 else False,
                                  cache_images=False if args.prebias else args.cache_images)
    # [I addded this line]Dataloader - The length of the loader will adapt to the batch_size.
    # so if your train dataset has 1000 samples and you use a batch_size of 10, the loader will have the length 100.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 16]),
                                             shuffle=not args.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = args.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    
    # print("\n\n\nDATALOADER", len(dataloader), "\n\n\n")

    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Starting %s for %g epochs...' % ('prebias' if args.prebias else 'training', epochs))
    epoch = 0

    # print("\n\n\nSTART EPOCH", start_epoch," ", epochs," EPOCHS \n\n\n")
    # print("\n\n\nLENGTH", dataset.__len__(), "\n\n\n")

    # exit(1)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx
        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # print("\n\n\ni-->", i, "\n\n\n")

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Multi-Scale training
            if multi_scale:
                if ni / acc % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % acc == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        plotter.plot("loss", "train", "Train YOLOV3", epoch, loss)

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if args.prebias:
            print_model_biases(model)
        else:
            # Calculate mAP (always test final epoch, skip first 10 if args.nosave)
            if epoch > 200:
                if not (args.notest or (args.nosave and epoch < 10)) or final_epoch:
                    with torch.no_grad():
                        results, maps = test.test(cfg_file,
                                                  data,
                                                  batch_size=batch_size,
                                                  img_size=args.img_size,
                                                  model=model,
                                                  conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                                  save_json=final_epoch and epoch > 0 and 'coco.data' in data)
        # Write epoch results
        with open(results_fl, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = results[2]  # mAP
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not args.nosave) or (final_epoch and not args.evolve) or args.prebias
        if save:
            with open(results_fl, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if args.bucket and not args.prebias:
                os.system('gsutil cp %s gs://%s' % (last, args.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights_dir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt
        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    if len(args.name):
        os.rename('results.txt', 'results_%s.txt' % args.name)
    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 8 *  = 16
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco_16img.data', help='*.data file path')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--epochs', type=int, default=273)   
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74

    args = parser.parse_args()
    args.weights = last if args.resume else args.weights
    print(args)
    device = torch_utils.select_device(args.device)

    print
    tb_writer = None
    if not args.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        run_prebias()  # optional
        train()  # train normally

    else:  # Evolve hyperparameters (optional)
        args.notest = True  # only test final epoch
        args.nosave = True  # only save final checkpoint
        if args.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % args.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                x = np.loadtxt('evolve.txt', ndmin=2)
                parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                if parent == 'single' or len(x) == 1:
                    x = x[fitness(x).argmax()]
                elif parent == 'weighted':  # weighted combination
                    n = min(10, x.shape[0])  # number to merge
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

                # Mutate
                np.random.seed(int(time.time()))
                s = [.2, .2, .2, .2, .2, .2, .2, .0, .02, .2, .2, .2, .2, .2, .2, .2, .2, .2]  # sigmas
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            run_prebias()
            results = train()

            # Write mutation results
            print_mutation(hyp, results, args.bucket)