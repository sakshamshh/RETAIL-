import os
import sys
import time
import logging
import subprocess
import threading

class OTAUpdater:
    """
    Over-The-Air Update Engine for the Edge Device.
    Periodically checks the remote Git repository for updates.
    If an update is found, it pulls the changes and gracefully restarts the Python process.
    """
    def __init__(self, check_interval_seconds=3600, logger=None):
        self.check_interval = check_interval_seconds
        self.logger = logger or logging.getLogger(__name__)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        self.logger.info(f"[OTA Updater] Started background Git watcher (interval: {self.check_interval}s)")

    def stop(self):
        self.running = False

    def _update_loop(self):
        # Give the main app some time to start up before checking for updates
        time.sleep(10)
        
        while self.running:
            try:
                self.logger.debug("[OTA Updater] Checking for remote updates...")
                
                # Fetch latest commits from remote
                subprocess.run(["git", "fetch"], check=True, capture_output=True)
                
                # Check if we are behind the remote main branch
                status = subprocess.run(["git", "status", "-uno"], check=True, capture_output=True, text=True)
                
                if "Your branch is behind" in status.stdout:
                    self.logger.warning("[OTA Updater] New version detected! Initiating OTA Update...")
                    
                    # Pull the new code
                    pull = subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True, text=True)
                    self.logger.info(f"[OTA Updater] Git Pull Successful:\n{pull.stdout}")
                    
                    self.logger.warning("[OTA Updater] Restarting Edge Service to apply updates...")
                    time.sleep(2)  # Give logs a moment to flush
                    
                    # Gracefully restart the current Python process
                    os.execv(sys.executable, ['python'] + sys.argv)
                    
                else:
                    self.logger.debug("[OTA Updater] Edge system is up to date.")
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"[OTA Updater] Git operation failed: {e.stderr}")
            except Exception as e:
                self.logger.error(f"[OTA Updater] Unexpected error: {e}")
                
            # Wait for the next check interval
            time.sleep(self.check_interval)
