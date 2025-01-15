import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Get RECORD_ALL_AUDIO from environment variable
RECORD_ALL_AUDIO = os.getenv('RECORD_ALL_AUDIO', 'false').lower() == 'true'


class RecordingManager:
    def __init__(self):
        self.recording_dir = Path('./recordings')
        self.ensure_recording_directory()
        self.record_all_audio = RECORD_ALL_AUDIO  # Store it as instance variable
        logger.info(f"RecordingManager initialized with RECORD_ALL_AUDIO = {self.record_all_audio}")

    def ensure_recording_directory(self):
        """Create recordings directory if it doesn't exist"""
        self.recording_dir.mkdir(exist_ok=True)
        logger.info(f"Recording directory ensured at: {self.recording_dir.absolute()}")

    def get_recording_paths(self, peer_uuid: str) -> dict:
        """Generate file paths for all recording files"""
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%mm_%S_%f')
        # Get absolute path
        abs_recording_dir = self.recording_dir.absolute()
        logger.info(f"Creating recording files in: {abs_recording_dir}")
        return {
            'from_vg': abs_recording_dir / f'{peer_uuid}_rec_from_vg_{timestamp}.raw',
            'to_oai': abs_recording_dir / f'{peer_uuid}_rec_to_oai_{timestamp}.raw',
            'from_oai': abs_recording_dir / f'{peer_uuid}_rec_from_oai_{timestamp}.raw',
            'to_vg_1': abs_recording_dir / f'{peer_uuid}_rec_to_vg_1_{timestamp}.raw',
            'to_vg_2': abs_recording_dir / f'{peer_uuid}_rec_to_vg_2_{timestamp}.raw'
        }
    async def initialize_recording_files(self, peer_uuid: str) -> dict:
        """Initialize all recording files for a session"""
        if not self.record_all_audio:  # Use instance variable
            return {}

        recording_files = self.get_recording_paths(peer_uuid)

        for file_path in recording_files.values():
            try:
                # Create empty file
                file_path.touch()
                logger.info(f"Created recording file: {file_path}")
            except Exception as e:
                logger.error(f"Error creating recording file {file_path}: {e}")

        return recording_files

    @staticmethod
    async def write_audio(file_path: Path, audio_data: bytes):
        """Write audio data to file"""
        try:
            logger.info(f"Attempting to write {len(audio_data)} bytes to {file_path}")
            with open(file_path, 'ab') as f:
                f.write(audio_data)
            logger.info(f"Successfully wrote to {file_path}")
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")