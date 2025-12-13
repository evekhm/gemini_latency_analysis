import os
import logging
import functools
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, export
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

# Global tracer instance
_TRACER = None

def init_tracer(project_id: Optional[str] = None):
    """
    Initialize OpenTelemetry tracer with Cloud Trace exporter.
    
    Args:
        project_id: Google Cloud Project ID. If None, attempts to read from env or default.
    """
    global _TRACER
    
    if not project_id:
        project_id = os.getenv('PROJECT_ID')
        
    if not project_id:
        logging.warning("⚠️ Warning: PROJECT_ID not set, skipping OpenTelemetry setup")
        return

    try:
        provider = TracerProvider()
        processor = export.BatchSpanProcessor(
            CloudTraceSpanExporter(project_id=project_id)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        # Instrument libraries
        GoogleGenAiSdkInstrumentor().instrument()
        SQLite3Instrumentor().instrument()
        
        _TRACER = trace.get_tracer(__name__)
        logging.info(f"✓ OpenTelemetry tracing enabled for project: {project_id}")
        
    except Exception as e:
        logging.warning(f"⚠️ Warning: Failed to setup OpenTelemetry: {e}")

def get_tracer():
    """Get the global tracer instance."""
    global _TRACER
    if _TRACER is None:
        # Fallback to no-op tracer if not initialized
        return trace.get_tracer(__name__)
    return _TRACER

def trace_span(name_override=None):
    """
    Decorator to manually wrap functions in a trace span.
    
    Args:
        name_override: Optional name for the span. If None, uses module.function_name.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name_override or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise e
        return wrapper
    return decorator
