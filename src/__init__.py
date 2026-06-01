"""BE-HPG: Boundary-Enhanced Hard Prototype Generation for few-shot medical image segmentation.

Source package. Subpackages:
  data/    MSD download, NIfTI -> 2D slice preprocessing, manifest, episodic sampler
  models/  protonet, meta_reweight (Meta-RCNN-inspired), be_hpg, shared backbones
  losses/  bce, boundary (Sobel-GT weighted BCE)
  metrics/ dice, iou, surface (hd95/asd via medpy, both-non-empty rule)
  engine/  train loop, eval loop, checkpointing (resumable)
  utils/   config loader, seeding, aggregation, figures
"""
