import argparse


from models import *   
from sys import platform
from utils.datasets import *
from utils.utils import *



def detect(save_txt=False):
    img_size = args.img_size  # (height, width) =(320, 192) or (416, 256) or (608, 352)  
    out, source, weights, view_img = args.output, args.source, args.weights, args.view_img

    # Delete output folder and make new folder to output the detection image
    if os.path.exists(out):
        shutil.rmtree(out) #imported from datasets.py 
    os.makedirs(out)   

    device = torch_utils.select_device(args.device)

    # Set YOLOV3 model 
    model = YOLOV3(args.cfg, img_size)

    # Check if weights file is to be gotten online
    try_download(weights) #specific files names to test are defined in models.try_dowload

    # If the weights file ends in pt, it is in pytorch format
    # otherwise, it is an online weights file in darknet format
    if weights.endswith('.pt'):  
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:   
        _ = load_darknet_weights(model, weights)


    model.to(device).eval() # set model to evaluation mode
    

    # set the image loader, and load the classes and colors
    dataset = ImageLoader(source, img_size=img_size)
    classes_list = classes_load(parse_data_cfg(args.data)['names'])
    color_list = [[random.randint(0, 255) for i in range(3)] for class_ in range(len(classes_list))]

    ###### need to change variable names, and also flag names, but not the flag names so as to change
    #them in train and test as well and not the changes in the google doc?

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        for i, det in enumerate(non_max_suppression(pred, args.conf_thres, args.nms_thres)):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes_list[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    # Add bbox to image
                    label = '%s %.2f' % (classes_list[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=color_list[int(cls)])

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)


            # Save image detection results 
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                 
                vid_path = save_path

                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*args.fourcc), fps, (w, h))
               
                vid_writer.write(im0)

    
    print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        detect()
