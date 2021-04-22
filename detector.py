import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detect_defect import detect_how

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def training(train_set, input_dir, output_dir, resume_dir, mask_dir,class_name):
    print(mask_dir)
    print(train_set)
    print(input_dir)
    print(output_dir)
    register_coco_instances(train_set, {}, train_set + "train_coco.json", train_set)
    #register_coco_instances(test_set, {}, test_set + "test_coco.json",  test_set)

    cfg = get_cfg()
    ###################

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if resume_dir == '':
        pass
    else:
        #cfg.MODEL.WEIGHTS = os.path.join('1' + '/exp/' + 'model_final.pth')
        cfg.MODEL.WEIGHTS = os.path.join(class_name + '/exp/' + 'model_0004999.pth')
    #Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = ()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LearningRate
    cfg.SOLVER.MAX_ITER = 5000  #No. of iterations   
    #cfg.TEST.EVAL_PERIOD = 10 # No. of iterations after which the Validation Set is evaluated. 
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)  
    trainer.resume_or_load(resume=False)
    if resume_dir == '':
        trainer.train()

    #cfg.MODEL.WEIGHTS = os.path.join("./output/model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set the testing threshold for this model    
    predictor = DefaultPredictor(cfg)
    
    files = os.listdir(input_dir)
    for file_ in files:   
        if file_.endswith('.jpg') or file_.endswith('.png'):
            im = cv2.imread(os.path.join(input_dir, file_))
            outputs = predictor(im)
            # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            #origin = v.draw_instance_predictions()
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
           
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            cv2.imwrite(output_dir + file_, out.get_image()[:, :, ::-1])
            detect_how(im,outputs["instances"].to("cpu").pred_masks.numpy(), mask_dir,file_, output_dir)
    DatasetCatalog.remove(train_set)
    #DatasetCatalog.remove(test_set)