source activate TensorFlow

python __main__.py --model SLSR --is_train True  --epoch 10 --f1 ./DATABASE/DIV2K/*.png --f2 ./DATABASE/Set5/*.bmp

python __main__.py --model SLSR --is_train False

python __main__.py --model CKP --is_train True --epoch 10000 --f1 ./DATABASE/DIV2K/ --f2 ./DATABASE/DIV2K_BlurKernel/

python __main__.py --model CKP --is_train False

python __main__.py --model SrOp --is_train True --epoch 10 --f1 ./DATABASE/DIV2K/*.png --f2 ./DATABASE/Set5/*.bmp

python __main__.py --model SrOp --is_train False

