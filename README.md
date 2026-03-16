## Train
MKL_THREADING_LAYER=GNU python -m doy_prediction.train_tile_cnn --outputs-root ./outputs --feature-set ndvi_only --train-years 2019 2020 2021 2022 --test-years 2023 --save-dir ./models_output/harvest_cnn_trial --epochs 30 --batch-size 32 --device cuda --wandb --wandb-project DeepSatModels-harvest

## Prediction
python -m doy_prediction.predict_tile_cnn --inputs ./outputs/2023_AR --checkpoints ./models/harvest_cnn_trial --feature-set ndvi_only --out-csv ./predictions_rice.csv --crops Rice