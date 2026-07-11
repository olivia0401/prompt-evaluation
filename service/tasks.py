"""
Job enqueueing.

`enqueue_run` puts a run on the RQ queue when Redis is reachable; otherwise it
executes the run inline (synchronously) so the service is fully functional with
no broker — useful for local dev, tests and single-box deployments.

The RQ worker imports `perform_run` as its job function (see service.worker).
"""
from . import settings
from .db import session_scope
from .models import Run, RunStatus


def perform_run(run_id: str) -> dict:
    """RQ job entrypoint. Thin wrapper so the queue payload is just a run_id."""
    from .runner import execute_run

    return execute_run(run_id)


def _get_queue():
    from redis import Redis
    from rq import Queue

    conn = Redis.from_url(settings.REDIS_URL)
    return Queue(settings.RQ_QUEUE_NAME, connection=conn, default_timeout=settings.JOB_TIMEOUT)


def enqueue_run(run_id: str) -> str | None:
    """
    Schedule a run. Returns the RQ job id, or None when executed inline.

    Falls back to inline execution if Redis is unavailable or INLINE_JOBS is set.
    """
    if settings.redis_available():
        try:
            job = _get_queue().enqueue(perform_run, run_id, job_id=f"run-{run_id}")
            with session_scope() as s:
                run = s.get(Run, run_id)
                if run is not None:
                    run.job_id = job.get_id()
            return job.get_id()
        except Exception:
            # Broker hiccup — degrade to inline rather than dropping the run.
            pass

    # Inline path: run it now, in-process.
    perform_run(run_id)
    return None
