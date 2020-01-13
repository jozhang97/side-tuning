
model=$1
task=rgb
mkdir /mnt/hdd1/taskonomy/small/$task/$model
mkdir /tmp/${task}/$model
tar -xf /mnt/hdd2/taskonomy/${task}/${model}_${task}.tar -C /tmp/${task}/$model/
cd /root/feature_selector
python -m feature_selector.shrink_images with data_dir=/tmp/ folders_to_convert=\[\"$model\"\] save_dir=/mnt/hdd1/taskonomy/small
rm -rf /tmp/${task}/$model/${task}
rmdir /tmp/${task}/$model
