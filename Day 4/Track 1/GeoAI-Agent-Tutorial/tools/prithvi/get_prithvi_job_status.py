from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def get_prithvi_job_status(job_id: str) -> dict:
    """Check the status of a submitted Prithvi inference job.

    job_id is the S3 OutputLocation returned by run_prithvi_inference.
    Returns status: 'running' | 'finished' | 'failed'.

    SageMaker Async Inference writes the result JSON to OutputLocation on
    success, and a .error file to OutputLocation + '.error' on failure.
    """
    parsed = urlparse(job_id)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return {"status": "finished", "message": "Results are ready."}
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            return {"status": "failed", "message": f"S3 error: {e}"}

    # SageMaker writes a .error file when the invocation fails
    try:
        s3.head_object(Bucket=bucket, Key=key + ".error")
        return {"status": "failed", "message": "Inference job failed on the endpoint."}
    except ClientError:
        pass

    return {"status": "running", "message": "Job is still processing."}
