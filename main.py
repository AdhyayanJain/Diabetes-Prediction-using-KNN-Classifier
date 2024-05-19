import os
import sys
from src.utils.logger import logging
from src.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.model_training_pipeline import ModelTrainingPipeline
from src.pipelines.model_prediction_pipeline import ModelPredictionPipeline

if __name__ == '__main__':

    logging.info('starting main module..!!')
    
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()

    data_preparation = DataPreparationPipeline()
    data_preparation.main()

    model_training = ModelTrainingPipeline()
    model_training.main()


    model_prediction = ModelPredictionPipeline()
    model_prediction.main()

    logging.info('finised running main module..!!')

