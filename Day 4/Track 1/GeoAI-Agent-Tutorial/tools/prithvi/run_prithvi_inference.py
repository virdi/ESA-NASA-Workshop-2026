import json
import os
import uuid

import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
ASYNC_BUCKET = os.environ.get("SAGEMAKER_ASYNC_BUCKET", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def run_prithvi_inference(
    task_type: str,
    bbox: list[float],
    date: str | None = None,
    date_range: dict | None = None,
    dates: list[str] | None = None,
) -> dict:
    """Submit an async Prithvi-EO-2.0 inference job via SageMaker Async Inference.

    task_type: 'flood' | 'burn' | 'crop'
    bbox: [west, south, east, north]
    date: YYYY-MM-DD (flood/burn)
    date_range: {'start_date': ..., 'end_date': ...} (crop)
    dates: list of 3 YYYY-MM-DD strings with >=70-day gaps (crop)

    Returns job_id (S3 output location). Poll with get_prithvi_job_status.
    Requires SAGEMAKER_ENDPOINT_NAME and SAGEMAKER_ASYNC_BUCKET env vars.
    """
    if task_type not in ("flood", "burn", "crop"):
        return {
            "message": f"Unsupported task_type '{task_type}'. Must be flood, burn, or crop."
        }
    if task_type in ("flood", "burn") and not date:
        return {"message": f"'date' is required for {task_type} task."}
    if task_type == "crop" and (not date_range or not dates or len(dates) != 3):
        return {"message": "Crop task requires date_range and exactly 3 dates."}
    if not ENDPOINT_NAME or not ASYNC_BUCKET:
        return {
            "message": "SAGEMAKER_ENDPOINT_NAME and SAGEMAKER_ASYNC_BUCKET must be set."
        }

    payload = {
        "task_type": task_type,
        "bbox": bbox,
        "date": date,
        "date_range": date_range,
        "dates": dates,
    }

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        input_key = f"async-input/{uuid.uuid4()}.json"
        s3.put_object(
            Bucket=ASYNC_BUCKET,
            Key=input_key,
            Body=json.dumps(payload),
            ContentType="application/json",
        )

        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        response = runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=f"s3://{ASYNC_BUCKET}/{input_key}",
            ContentType="application/json",
        )

        return {
            "job_id": response["OutputLocation"],
            "status": "submitted",
            "message": "Job submitted to SageMaker Async endpoint.",
        }
    except ClientError as e:
        return {"status": "failed", "message": f"Job submission failed: {e}"}
