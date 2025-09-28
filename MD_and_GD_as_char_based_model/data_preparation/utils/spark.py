"""
Spark utility helpers.
"""

import pyspark.sql as ps  # type: ignore


class SparkSessionSingleton:
    """Manages unique Spark sessions per app name."""

    _SESSIONS = {}

    def __init__(self):
        raise NotImplementedError("SparkSessionSingleton should not be instantiated directly.")

    @classmethod
    def get(cls, app_name, params_func=None):
        """Retrieve or create a SparkSession and context for the given app name."""
        if app_name not in cls._SESSIONS:
            builder = ps.SparkSession.builder.appName(app_name)
            if params_func:
                params_func(builder)
            session = builder.getOrCreate()
            context = session.sparkContext
            context.setLogLevel("ERROR")
            cls._SESSIONS[app_name] = (session, context)
        return cls._SESSIONS[app_name]

    @classmethod
    def cleanup(cls):
        """Close all tracked Spark sessions."""
        for session, _ in cls._SESSIONS.values():
            session.stop()
        cls._SESSIONS = {}
