#!/bin/bash
curl "https://drive.usercontent.google.com/download?id=1zlYhpT1UtYeA3zbIAE1pNxIb9yZTrMbM&confirm=xxx" -o "test_dataset_scene_79.zip"

unzip test_dataset_scene_79.zip
rm -rf test_dataset_scene_79.zip
mv test_dataset_scene scenes_evalCaption