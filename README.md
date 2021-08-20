# face_of_manga
The final paper is [here](https://github.com/jennifer66666/face_of_manga/blob/revise/final_paper.pdf)  
The image dataset should be downloaded from [Manga109](http://www.manga109.org/en/)  
The bounding box of faces should be cropped as [Chu et.al.](https://www.cs.ccu.edu.tw/~wtchu/projects/MangaFace/)  
The manually labeled landmarks can be found in [oaugereau](https://github.com/oaugereau/FacialLandmarkManga)  

*To run the training  

```

python3  face-of-art/train_heatmaps_network.py --output_dir='test_global' --augment_geom=True   
--augment_texture=False --p_texture=1. --p_geom=1. --img_path  dataset   --train_crop_dir crop_gt_margin_0.25_train  
--no_need_bb True --valid_data crop_gt_margin_0.25_val --num_landmarks 60 --global_feature

```
*To run the inferencing 
```
python3 face-of-art/predict_landmarks.py  
```
