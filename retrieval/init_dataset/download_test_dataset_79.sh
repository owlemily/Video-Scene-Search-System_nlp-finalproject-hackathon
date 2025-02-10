!/bin/bash
curl "https://drive.usercontent.google.com/download?id=1CKTgRg_iQFT_yi2THsuyM-JSXS1kHfRj&confirm=xxx" -o "test_dataset_79.zip"

unzip test_dataset_79.zip
rm -rf test_dataset_79.zip
rm -rf __MACOSX

mkdir -p ../datasets
mv test_dataset_79 ../datasets/