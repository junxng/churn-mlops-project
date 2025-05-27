import os
from src.Churn.utils.logging import logger
from src.Churn.pipeline.kubeflow import pipeline
from kfp import Client
from kfp.compiler import Compiler
Compiler().compile(
    pipeline_func=pipeline,
    package_path="sentiment_pipeline.yaml"
)

client = Client(host="http://your-kubeflow-endpoint")
client.create_run_from_pipeline_package(
    pipeline_file="sentiment_pipeline.yaml",
    arguments={}
)