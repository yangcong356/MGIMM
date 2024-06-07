from typing import Dict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocoevalcap.eval import COCOEvalCap
import json


if __name__ == "__main__":
    pred_path = '/data1/users/yangcong/code/MultimodalforLongCaption/BMGPG/scripts/eval/BMGPG-phi2-2.3b-lora-r64-b1-e5-dior-gpg-answer-coco.json'
    ref_gt_path = '/data1/users/yangcong/data/GPG/annotations/dior-gpg/coco-dior-gpg-test.json'

    # create coco object and coco_result object
    coco = COCO(ref_gt_path)
    coco_result = coco.loadRes(pred_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
