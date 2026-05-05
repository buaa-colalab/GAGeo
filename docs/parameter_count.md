总表

  | Model | Total | Trainable |
  |---|---:|---:|
  | gageo-dino-vit_b | 491.5M | 182.4M |
  | gageo-dino-vit_h | 1103.4M | 794.3M |
  | gageo | 937.4M | 628.3M |
  | trogeo | 71.3M | 71.3M |
  | trogeo-pi3 | 781.9M | 781.9M |

  GAGeo-DINO-ViT-B

  | Module | Total | Trainable |
  |---|---:|---:|
  | backbone | 392.537M | 88.165M |
  | prompt_encoder | 0.404M | 0.398M |
  | bbox_head | 34.634M | 34.634M |
  | mask_head | 14.950M | 14.950M |
  | heatmap_head | 0.002M | 0.002M |
  | camera_head | 39.514M | 39.514M |
  | contrastive_head | 9.442M | 4.721M |

  backbone 细分：

  | Submodule | Total | Trainable |
  |---|---:|---:|
  | encoder | 304.372M | 0 |
  | decoder (ViT-B fusion blocks) | 85.054M | 85.054M |
  | vit_norm | 0.002M | 0.002M |
  | dino_to_vit | 0.787M | 0.787M |
  | prompt_coord_mlp | 0.593M | 0.593M |
  | final_proj | 1.575M | 1.575M |

  GAGeo-DINO-ViT-H

  | Module | Total | Trainable |
  |---|---:|---:|
  | backbone | 939.965M | 635.594M |
  | prompt_encoder | 0.669M | 0.663M |
  | bbox_head | 34.634M | 34.634M |
  | mask_head | 14.950M | 14.950M |
  | heatmap_head | 0.002M | 0.002M |
  | camera_head | 103.773M | 103.773M |
  | contrastive_head | 9.442M | 4.721M |

  backbone 细分：

  | Submodule | Total | Trainable |
  |---|---:|---:|
  | encoder | 304.372M | 0 |
  | decoder (ViT-H fusion blocks) | 629.678M | 629.678M |
  | vit_norm | 0.003M | 0.003M |
  | dino_to_vit | 1.312M | 1.312M |
  | prompt_coord_mlp | 1.644M | 1.644M |
  | final_proj | 2.623M | 2.623M |

  GAGeo（我们的模型）

  | Module | Total | Trainable |
  |---|---:|---:|
  | backbone | 766.323M | 461.951M |
  | prompt_encoder | 0.537M | 0.530M |
  | bbox_head | 8.403M | 8.403M |
  | mask_head | 14.950M | 14.950M |
  | heatmap_head | 0.002M | 0.002M |
  | camera_head | 67.712M | 67.712M |
  | contrastive_head | 9.442M | 4.721M |
  | inter_bbox_heads | 25.209M | 25.209M |
  | inter_mask_heads | 44.850M | 44.850M |

  backbone 细分：

  | Submodule | Total | Trainable |
  |---|---:|---:|
  | encoder | 304.372M | 0 |
  | decoder (Pi3 decoder) | 453.547M | 453.547M |
  | intermediate_projs | 6.298M | 6.298M |
  | final_proj | 2.099M | 2.099M |

  TROGeo

  | Module | Total | Trainable |
  |---|---:|---:|
  | query_model (Swin-S) | 48.836M | 48.836M |
  | combine_clickptns_conv | 0.000M | 0.000M |
  | cross_attention | 12.992M | 12.992M |
  | fcn_out | 4.736M | 4.736M |
  | coodrs_out | 4.719M | 4.719M |

  说明：reference_model 和 query_model 共享同一套 Swin-S 权重，所以不会再额外加一份参数。

  TROGeo-π³

  | Module | Total | Trainable |
  |---|---:|---:|
  | backbone_adapter | 759.497M | 759.497M |
  | combine_clickptns_conv | 0.000M | 0.000M |
  | cross_attention | 12.992M | 12.992M |
  | fcn_out | 4.736M | 4.736M |
  | coodrs_out | 4.719M | 4.719M |

  backbone_adapter 细分：

  | Submodule | Total | Trainable |
  |---|---:|---:|
  | proj | 1.574M | 1.574M |
  | encoder | 304.372M | 304.372M |
  | decoder | 453.547M | 453.547M |