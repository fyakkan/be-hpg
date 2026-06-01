"""Models (Phase 2).

Planned modules:
  backbones.py     timm backbones (mobilevit_s, deit_tiny_patch16_224, resnet50) ->
                   dense feature maps + 1x1 projection to embed_dim, L2-normalized
  protonet.py      Prototypical Network (masked GAP prototype, cosine matching)
  meta_reweight.py Meta-RCNN-inspired support-conditioned channel reweighting + conv head
  be_hpg.py        Proposed: SSP boundary-band hard prototypes (mini-batch k-means),
                   cosine matching, 1x1 conv/sigmoid head
"""
