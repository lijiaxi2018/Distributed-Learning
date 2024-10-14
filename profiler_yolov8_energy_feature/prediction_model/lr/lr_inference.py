import joblib

def linear_regression_inference(iteration):
    # Load the model from the file
    loaded_lr = joblib.load('/home/jiaxi/cs525/Experiment/prediction_model/lr/linear_regression_model.pkl')

    sample = [[5.0, 3.6, 1.4, 0.2]]
    for i in range(iteration):
        loaded_lr.predict(sample)