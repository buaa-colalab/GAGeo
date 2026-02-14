import torch
from models.cross_view_localizer_v2 import CrossViewLocalizerV2

model = CrossViewLocalizerV2(
    img_size=518,
    patch_size=14,
    decoder_size='large',
    num_learnable_tokens=2,
    supervision_layers=[4, 11, 17],
    supervision_weights=[0.1, 0.3, 0.6],
    contrastive=True,
    sam_embed_dim=256,
    num_mask_tokens=1,
)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print('TOTAL', nparams(model), 'TRAINABLE', trainable(model))
for name in [
    'backbone',
    'prompt_encoder',
    'bbox_head',
    'mask_head',
    'heatmap_head',
    'camera_head',
    'contrastive_head',
    'inter_bbox_heads',
    'inter_mask_heads',
]:
    mod = getattr(model, name)
    print(name, nparams(mod), trainable(mod))

bb = model.backbone
print('dec_depth', bb.dec_depth, 'num_stage_layers', bb.num_stage_layers, 'dec_embed_dim', bb.dec_embed_dim, 'output_dim', bb.output_dim)
print('num_decoder_blocks', len(bb.decoder), 'num_masked_blocks', len(bb.masked_blocks))
print('num_supervision_proj', len(bb.intermediate_projs))
print('inter_bbox_heads keys', list(model.inter_bbox_heads.keys()))
print('inter_mask_heads keys', list(model.inter_mask_heads.keys()))