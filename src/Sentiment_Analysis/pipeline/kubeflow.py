from kfp.components import load_component_from_file
import kfp.dsl as dsl
# ingest_component.py
@dsl.component
def data_ingestion_component() -> str:
    import os
    from Sentiment_Analysis.pipeline.prepare_data import DataPreparationPipeline
    pipeline = DataPreparationPipeline()
    pipeline.main()
    return "Data prepared"
# full_workflow_component.py
@dsl.component
def full_workflow_component() -> str:
    from Sentiment_Analysis.pipeline.main_pipeline import WorkflowRunner
    runner = WorkflowRunner()
    runner.run()
    return "Workflow completed"

    
    
# pipeline.py
@dsl.pipeline(name="SENTIMENT_ANALYSIS", description="Data Ingestion -> Model Prep -> Train/Eval")
def sentiment_pipeline():
    ingest_task = data_ingestion_component()
    workflow_task = full_workflow_component()
    workflow_task.after(ingest_task) 

