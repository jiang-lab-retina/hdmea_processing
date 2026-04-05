
Drive list

L:	\\Jiangfs1\fs_1_1_data
M:	\\Jiangfs1\fs_1_2_data
Q:	\\Jiangfs2\fs_2_1_data
R:	\\Jiangfs2\fs_2_2_data
O:	\\Jiangfs3\fs_3_1_data
P:	\\Jiangfs3\fs_3_2_data
S:	\\Jiangfs4\fs_4_1_data
T:	\\Jiangfs4\fs_4_2_data

1. write script to get the columns from /m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/natural_scene/file_name_natural_scene.xlsx

2. search the cmcr and cmtr files in the drive list and get their absolute path use Drive name (not UNC path). Note there is a different from file_name to real file name in usage of ".", "-" and others. For example the real file name of 2025.02.04.13.22.13.Rec.cmcr is 2025.02.04-13.22.13-Rec.cmcr

3. convert the cmtr file to h5 according to the protocol column: 
use /m:/Python_Project/Data_Processing_2027/Projects/unified_pipeline/batch_all_steps.py for file with protocol play_all_optimization_set6_manual_ipRGC()

4. for file with protocol play_natural_scene_movie_v1(): use the basic information and section timing part of /m:/Python_Project/Data_Processing_2027/Projects/unified_pipeline/batch_all_steps.py

5. limit the output data and code within the folder of /m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/natural_scene