import joblib

def random_forest_inference(iteration):
    # Load the model from the file
    loaded_rf = joblib.load('/home/jiaxi/cs525/Experiment/prediction_model/rf/random_forest_model.joblib')

    sample = [[5.0, 3.6, 1.4, 0.2]]
    for i in range(iteration):
        loaded_rf.predict(sample)