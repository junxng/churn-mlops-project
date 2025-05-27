from kfp.components import load_component_from_file
import kfp.dsl as dsl
# ingest_component.py

# full_workflow_component.py
@dsl.component
def full_workflow_component() -> str:
    from Churn.pipeline.main_pipeline import WorkflowRunner
    runner = WorkflowRunner()
    runner.run()
    return "Workflow completed"

    
    
# pipeline.py
@dsl.pipeline(name="SENTIMENT_ANALYSIS", description="Data Ingestion -> Model Prep -> Train/Eval")
def sentiment_pipeline():
    workflow_task = full_workflow_component()

