"""Base pipeline class"""

import os
from datetime import datetime
from .utils import safe_print


class BasePipeline:
    """Base class for pipelines with common functionality"""

    def __init__(self, config):
        self.cfg = config
        self.log_fh = None
        self.artifacts = {
            "active_steps": [],
            "pool": {}
        }

        # Setup logging
        self.setup_logger()

        # Generate run ID if not provided
        if not hasattr(self.cfg, 'run_id') or not self.cfg.run_id:
            self.cfg.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    def setup_logger(self):
        """Setup logging to file"""
        if hasattr(self.cfg, 'output_folder') and self.cfg.output_folder:
            os.makedirs(self.cfg.output_folder, exist_ok=True)

            if hasattr(self.cfg, 'log_to_file') and self.cfg.log_to_file:
                log_path = os.path.join(
                    self.cfg.output_folder,
                    f"pipeline_log_{self.cfg.run_id}.txt"
                )
                try:
                    self.log_fh = open(log_path, "w", encoding="utf-8")
                except Exception:
                    self.log_fh = None

    def _log(self, msg: str):
        """Log message to console and file"""
        safe_print(msg)

        if self.log_fh:
            try:
                self.log_fh.write(msg + "\n")
                self.log_fh.flush()
            except Exception:
                pass

    def _activate(self, step_name: str):
        """Mark a step as active"""
        self.artifacts["active_steps"].append(step_name)

    def close(self):
        """Clean up resources"""
        if self.log_fh:
            try:
                self.log_fh.close()
            except Exception:
                pass
            self.log_fh = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
