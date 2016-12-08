# chainer-pose-hg
Chainer implementation of Stacked Hourglass Networks for Human Pose Estimation

## Requirement
- Python 2.7
- Chainer

## Prepare data
- [LSP dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip)
- [LSP Extended dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip)

Place all images in data/LSP/images

## Start training
```
python src/train.py
```
Check the options by `python src/train.py --help` and modify the training settings.

## Get predictions
```
python src/test.py
```
Default setting is for predicting LSP test set(1000 images).

Also, you can predict any image by specifying image name in an csv file,
set --test_csv_fn and --img_dir

## Pre-train model
Will be uploaded soon

## Note
Current implementation uses 2 stacks of hourglass and 1 residual modules at each location

refer to the '-nStack' and '-nModules' in opts.lua of [pose-hg-train](https://github.com/anewell/pose-hg-train)

## Reference
- [pose-hg-train](https://github.com/anewell/pose-hg-train)
- http://seiya-kumada.blogspot.tw/2016/03/fully-convolutional-networks-chainer.html
- [DeepPose](https://github.com/mitmul/deeppose)
