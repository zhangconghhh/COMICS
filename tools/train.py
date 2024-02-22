# -*- coding: UTF-8 -*-
from turtle import pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import logging, pdb
import os, shutil
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DATASET_ROOT = 'datasets/'
ANN_ROOT = os.path.join(DATASET_ROOT, 'Images/annotations')
TRAIN_JSON = os.path.join(ANN_ROOT, 'Train_poly.json')
VAL_JSON = os.path.join(ANN_ROOT, 'Val_poly.json')

DATASET_CATEGORIES = [
    {"name": "Real", "id": 0, "isthing": 1, "color": [220, 20, 60]},
    {"name": "Fake", "id": 1, "isthing": 1, "color": [219, 142, 185]},
]

PREDEFINED_SPLITS_DATASET = {
    "train_2022": (DATASET_ROOT, TRAIN_JSON),
    "val_2022": (DATASET_ROOT, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)






class Trainer(DefaultTrainer):
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                # 2->22046  4->11020  1->44097   8->5510  12->3672  32->1376  16->2754
                epoch = int(self.iter/5510) #22046 5510; 
                self.model.set_con_avliable(min(max(0, (epoch-6)/5), 1.0)*0.5) 
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON: #flase
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() 
    cfg.set_new_allowed(True) 
    cfg.merge_from_file(args.config_file)  
    cfg.merge_from_list(args.opts)      
    cfg.DATASETS.TRAIN = ("train_2022",)
    cfg.DATASETS.TEST = ("val_2022",)
    cfg.DATALOADER.NUM_WORKERS = 0  
 
    # Imgsize = 512
    Imgsize = 1024
    cfg.INPUT.MAX_SIZE_TRAIN = Imgsize  
    cfg.INPUT.MAX_SIZE_TEST = Imgsize 
    cfg.INPUT.MIN_SIZE_TRAIN = (Imgsize, Imgsize) 
    cfg.INPUT.MIN_SIZE_TEST = Imgsize
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES 
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.BASIS_MODULE.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.FCOS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MEInst.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.SOLOV2.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.WEIGHTS = "R-50.pkl"    
    cfg.SOLVER.IMS_PER_BATCH = 8 
    cfg.SOLVER.BASE_LR=0.01 
    TOTAL_IMAGE = 44097
    ITERS_IN_ONE_EPOCH = int(TOTAL_IMAGE / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs
    cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH*8-1,ITERS_IN_ONE_EPOCH*11-1) 
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.IS_ROTATE = True
    cfg.freeze()
    default_setup(cfg, args)

    return cfg



def backup_code(cfg):
    
    if os.path.exists(cfg.OUTPUT_DIR+'/fcos'):
        shutil.rmtree(cfg.OUTPUT_DIR+'/fcos')
    if os.path.exists(cfg.OUTPUT_DIR+'/blendmask'):
        shutil.rmtree(cfg.OUTPUT_DIR+'/blendmask')

    shutil.copytree('adet/modeling/fcos/', cfg.OUTPUT_DIR +'/fcos')
    shutil.copytree('adet/modeling/blendmask/', cfg.OUTPUT_DIR+'/blendmask')
    shutil.copy('tools/train.py', cfg.OUTPUT_DIR+'/train.py')
    shutil.copy('adet/data/dataset_mapper.py', cfg.OUTPUT_DIR+'/dataset_mapper.py')


def main(args):
    
    cfg = setup(args)
    register_dataset()
    backup_code(cfg)

   
    if args.eval_only:
    # if True:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """   
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
   
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()



if __name__ == "__main__":

    args = default_argument_parser().parse_args()
   
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
