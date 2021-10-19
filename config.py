# =============================================================================
# GazeFollow dataset dir config
# =============================================================================
gazefollow_train_data = "data/gazefollow"
gazefollow_train_depth = "data/gazefollow/train_depth_1_with_norm"
gazefollow_train_label = "data/gazefollow/train_annotations_release.txt"
gazefollow_val_data = "data/gazefollow"
gazefollow_val_depth = "data/gazefollow/test2_depth_1_with_norm"
gazefollow_val_label = "data/gazefollow/test_annotations_release.txt"


# =============================================================================
# VideoAttTarget dataset dir config
# =============================================================================
videoattentiontarget_train_data = "data/videoatttarget/images"
videoattentiontarget_train_label = "data/videoatttarget/annotations/train"
videoattentiontarget_val_data = "data/videoatttarget/images"
videoattentiontarget_val_label = "data/videoatttarget/annotations/test"


# =============================================================================
# model config
# =============================================================================
input_resolution = 224
output_resolution = 64
angle_heatmap_width = 180
angle_heatmap_heigh = 180
