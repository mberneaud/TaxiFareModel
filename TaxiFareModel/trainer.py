# imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient


class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[GER] [Berlin] [mberneaud] test_model v2"  # 🚨 replace with your country code, city, github_nickname and model name and version

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.fitted_pipe = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # feat engineering
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())

        # Column selection
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        # Preprocessing pipeline
        feat_eng_bloc = ColumnTransformer([
            ('time', pipe_time, time_cols),
            ('distance', pipe_distance, dist_cols)
        ])  # remainder='passthrough'

        # model including the pipeline set as class attribute
        self.pipeline = Pipeline(
            steps=[('feat_eng_bloc',
                    feat_eng_bloc), ('regressor', RandomForestRegressor())])
        return self.pipeline

    def run(self, **kwargs):
        """set and train the pipeline"""
        self.fitted_pipe = self.set_pipeline()
        self.fitted_pipe.fit(self.X, self.y, **kwargs)
        return self.fitted_pipe


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.fitted_pipe.predict(X_test)
        self.model_score = compute_rmse(y_pred, y_test)
        return self.model_score

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":
    data = get_data()  # get data
    clean_data = clean_data(data)  # clean data
    X = clean_data.drop(columns="fare_amount") # Creating X
    y = clean_data["fare_amount"] # and y
    X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2) # Creating a holdout
    trainer = Trainer(X_train, y_train)  # hold out set is created during class instantiation
    trainer.set_pipeline()  # pipeline creation
    trainer.run()  # running, model stored in instance
    score = trainer.evaluate(X_test, y_test)# evaluate
    trainer.mlflow_log_param("model", "random forest")
    trainer.mlflow_log_metric("rmse", score)
    print(f"the model rmse is:{score:.4f}")
