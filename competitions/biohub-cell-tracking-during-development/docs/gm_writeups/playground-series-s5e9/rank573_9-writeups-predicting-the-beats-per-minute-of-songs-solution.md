# Predicting-the-beats-per-minute-of-songs-solution-private-lb-postion-573

🎵 Predicting the Beats-per-Minute (BPM) of Songs
📌 Competition Overview
- 
The task is to predict the BeatsPerMinute (BPM) of songs using features derived from audio and metadata.
This challenge tests skills in feature engineering, regression modeling, and ensemble learning.

📂 **Datasets Used**
   Playground-series-s5e9 dataset
   Original dataset (bpm-prediction-challenge)

🔄 **Preprocessing**

     Dropped missing values in original dataset. 
    Concatenated with Playground training data.

    Removed duplicates.

    Defined:
         X = Features
         y = Target (BeatsPerMinute)
         X_test = Test Features

🛠️ **Feature Engineering**
       A wide range of domain-inspired features were created to improve model accuracy:
       Basic Interactions: RhythmEnergy, MoodEnergy, LoudnessEnergy
       Music Theory Features: TempoStability, DynamicIntensity, VocalInstrumentBalance
       Time-based: DurationMinutes, BeatsPerSecond, EnergyDensity
       Statistical: AudioFeatureMean, AudioFeatureStd, AudioFeatureRange
       Custom: Danceability, HighEnergy, HighRhythm
     Transformations: Log & Power transforms for skewed features

⚙️ **Feature Processing**
      Applied Yeo-Johnson PowerTransformer to reduce skewness.
      Used RobustScaler to handle outliers.

🔍 Feature Selection
Initial LightGBM model trained on full features.
Top 20 most important features were selected, including:
    ['LivePerformanceLikelihood' 'VocalAcoustic' 'RhythmDuration'
    'InstrumentalLive' 'VocalContent' 'LoudnessEnergyVocal' 'AcousticQuality'
     'VocalInstrumentBalance' 'MoodToRhythm' 'RhythmDensity' 'AudioFeatureMax'
     'EnergyDensity' 'LoudnessRhythm' 'MoodScore' 'TrackDurationMs'
     'NormRhythmScore' 'Danceability' 'EnergyDuration' 'RhythmEnergyDiff'
     'MoodEnergy']

🚀 **Modeling Approach**

The final model is based on LightGBM Regression with optimized hyperparameters.
Parameters:
   n_estimators = 10000
   learning_rate = 0.008 
   num_leaves = 64
   max_depth = 8
   Regularization: reg_alpha = 0.5, reg_lambda = 0.5
   Sampling: subsample = 0.7, colsample_bytree = 0.6

Cross Validation:
         10-Fold K-Fold CV used for stability.
         Early stopping enabled.

📜 Training Logs
  Training final model with selected features...
  Fold 1 RMSE: 3.05
  Fold 2 RMSE: 3.12
  Fold 3 RMSE: 3.08
  Fold 4 RMSE: 3.14
  Fold 5 RMSE: 3.09
  Fold 6 RMSE: 3.11
   Fold 7 RMSE: 3.04
   Fold 8 RMSE: 3.15
   Fold 9 RMSE: 3.07
   Fold 10 RMSE: 3.12

Average RMSE: 3.10 (±0.23)

📈 Results

Private Score: 26.40632
Best Score: 26.40632 (V5)

Predictions were clipped between 60–200 BPM to ensure realism.

✅ Conclusion
This solution combines:
   Advanced feature engineering
   Feature selection based on LightGBM importance
   Robust scaling + transformation
   10-fold cross-validation with LightGBM
  🎯 The final model provides  BPM predictions with stable performance across folds.

🙌 Acknowledgments
       Kaggle for hosting the challenge
       BPM Prediction Challenge dataset creators

✨ Footer
It is First write ups i am bit confused whether i am writing good so I want to anyone who has better suggestion about writing proper writeup 
and improvements and if i am missing something in writing you can comments Thanks for Advance.
Thanks for reading 💙 — if you found this notebook helpful, don’t forget to ⭐ upvote and share your thoughts in the comments! 🚀