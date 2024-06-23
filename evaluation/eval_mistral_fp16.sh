set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export INFERENCE_RAM=4

for seminar in 8 9 10 11 12; do
    python generate_questions.py ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf ../data/ru/"$seminar".pdf
done
