## ECG Heartbeat Classification

Code to a project related to the 2020 ETHZ course "Machine Learning for Health Care" by Gunnar Rätsch.

* Please make sure to include the data in the correct directory, e.g. for ECG_Heartbeat_Classification, put the `.csv` files in `data/ECG_Heartbeat_Classification`.

* Depending if you run on CPU or GPU, choose the conda environments `ml_env.yml` or `tf_gpu.yml` respectively. 

* For all results, run `run_all.ipyn`. If some parts fail (they shouldn't), run the two scripts in `data_exploration` and `experiments` manually. 

Important note: For transfer learning, make sure that the new model is trained on the same device type as the base model! (e.g. CPU/CPU or GPU/GPU)

