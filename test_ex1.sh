echo 'set5' &&
echo '0.2' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set5/HR"  --ratio 0.2 --dataset_name 'set5'&&
echo '0.4' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set5/HR"  --ratio 0.4 --dataset_name 'set5'&&
echo '0.6' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set5/HR"  --ratio 0.6 --dataset_name 'set5'&&
echo '0.8' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set5/HR"  --ratio 0.8 --dataset_name 'set5'&&


echo 'set14' &&
echo '0.2' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set14/HR"  --ratio 0.2 --dataset_name 'set14'&&
echo '0.4' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set14/HR"  --ratio 0.4 --dataset_name 'set14'&&
echo '0.6' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set14/HR"  --ratio 0.6 --dataset_name 'set14'&&
echo '0.8' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Set14/HR"  --ratio 0.8 --dataset_name 'set14'&&

echo 'bsd100' &&
echo '0.2' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/B100/HR" --ratio 0.2 --dataset_name 'bsd100'&&
echo '0.4' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/B100/HR" --ratio 0.4 --dataset_name 'bsd100'&&
echo '0.6' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/B100/HR" --ratio 0.6 --dataset_name 'bsd100'&&
echo '0.8' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/B100/HR" --ratio 0.8 --dataset_name 'bsd100'&&

echo 'urban100' &&
echo '0.2' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Urban100/HR" --ratio 0.2 --dataset_name 'urban100'&&
echo '0.4' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Urban100/HR" --ratio 0.4 --dataset_name 'urban100'&&
echo '0.6' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Urban100/HR" --ratio 0.6 --dataset_name 'urban100'&&
echo '0.8' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/benchmark/Urban100/HR" --ratio 0.8 --dataset_name 'urban100'&&

echo 'div2k' &&
echo '0.2' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/div2k/DIV2K_valid_HR" --ratio 0.2 --dataset_name 'div2k_val'&&
echo '0.4' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/div2k/DIV2K_valid_HR" --ratio 0.4 --dataset_name 'div2k_val'&&
echo '0.6' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/div2k/DIV2K_valid_HR" --ratio 0.6 --dataset_name 'div2k_val'&&
echo '0.8' &&
python test-irregular_ex1.py --root "/home/ylwang/dataset/div2k/DIV2K_valid_HR" --ratio 0.8 --dataset_name 'div2k_val'&&


true
