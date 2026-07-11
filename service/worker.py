"""
RQ worker entrypoint.

    python -m service.worker            # start a worker listening on the evals queue

Requires Redis (REDIS_URL). The worker imports service.tasks.perform_run as the
job function, which drives service.runner.execute_run.
"""
from redis import Redis
from rq import Queue, Worker

from . import settings
from .db import init_db


def main() -> None:
    init_db()  # ensure tables exist before the first job writes
    conn = Redis.from_url(settings.REDIS_URL)
    queue = Queue(settings.RQ_QUEUE_NAME, connection=conn, default_timeout=settings.JOB_TIMEOUT)
    worker = Worker([queue], connection=conn)
    print(f"[worker] listening on '{settings.RQ_QUEUE_NAME}' at {settings.REDIS_URL}")
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
