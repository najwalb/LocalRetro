# Scripts

1. Create debug subset (500 rows from train/val)
  python LocalRetro/scripts/create_debug_subset.py

2. Extract templates from debug training data
  python LocalRetro/preprocessing/Extract_from_train_data.py -d USPTO_FULL_debug

3. Label reactions with templates
  python LocalRetro/preprocessing/Run_preprocessing.py -d USPTO_FULL_debug

4. Train 2 epochs on debug subset
  python LocalRetro/scripts/Train.py -d USPTO_FULL_debug -b 16 -n 2 --overwrite

  All scripts use absolute PROJECT_ROOT paths, so you can run them from any directory.


# Create conda env

conda create -c conda-forge -n localretro python=3.7 -y                                                                                                                                                     │
│ conda activate localretro                                                                                                                                                                                   │
│ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y                                                                                                                      │
│ conda install -c conda-forge rdkit -y                                                                                                                                                                       │
│ pip install dgl dgllife scikit-learn numpy pandas 