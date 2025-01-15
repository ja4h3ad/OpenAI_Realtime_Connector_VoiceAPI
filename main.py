from fastapi import FastAPI, WebSocket, Query, Request
import logging
import json
import os
from dotenv import load_dotenv
import asyncio
import websockets
import base64
import numpy as np
import time
# below imports needed for down- and upsampling
from scipy.signal import resample_poly
from typing import Union
from datetime import datetime
from fastapi.websockets import WebSocketDisconnect
from pathlib import Path

# Setup
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()

# Core configuration for secrets and env variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT')  # should be "api.openai.com/v1/realtime"
OPENAI_MODEL = os.getenv('OPENAI_MODEL')        # e.g. "gpt-4o-realtime-preview"
OPENAI_VOICE_NAME = os.getenv('OPENAI_VOICE_NAME')
OPENAI_SYSTEM_MESSAGE = os.getenv('OPENAI_SYSTEM_MESSAGE')
OPENAI_INPUT_AUDIO_TRANSCRIPTION = os.getenv('OPENAI_INPUT_AUDIO_TRANSCRIPTION')
OPENAI_TURN_DETECTION_TYPE = os.getenv('OPENAI_TURN_DETECTION_TYPE')

# OpenAI Session Settings for Realtime API
OPENAI_SESSION_SETTINGS = {
    "type": "session.update",
    "session": {
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {
            "type": OPENAI_TURN_DETECTION_TYPE,
            "silence_duration_ms": 300
        },
        "input_audio_transcription": {
            "model": OPENAI_INPUT_AUDIO_TRANSCRIPTION
        },
        "voice": OPENAI_VOICE_NAME,
        "instructions": OPENAI_SYSTEM_MESSAGE,
        "modalities": ["text", "audio"],
        "temperature": 0.8
    }
}

# OpenAI websocket construct, notice the model is part of query params
OPENAI_WS_URL = f"wss://{OPENAI_ENDPOINT}?model={OPENAI_MODEL}"

# Set up headers in the format OpenAI expects
OPENAI_WS_HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

# Audio resampling functions
def downsample_audio(input_buffer: Union[bytes, np.ndarray],
                     input_sample_rate: int = 24000,
                     output_sample_rate: int = 16000) -> bytes:
    """
    Downsample audio using scipy's polyphase resampling.
    24kHz to 16kHz (ratio 2:3); needed for PSTN connection
    """
    try:
        # Convert input buffer to numpy array if needed
        if isinstance(input_buffer, bytes):
            input_array = np.frombuffer(input_buffer, dtype=np.int16)
        else:
            input_array = input_buffer

        # Resample using polyphase filter (2:3 ratio for 24kHz to 16kHz)
        resampled_float = resample_poly(input_array.astype(np.float32), 2, 3)

        # Convert back to int16 with proper clipping
        output_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return output_int16.tobytes()

    except Exception as e:
        logger.error(f'Error during downsampling rate conversion: {e}')
        raise


def upsample_audio(input_buffer: Union[bytes, np.ndarray],
                   input_sample_rate: int = 16000,
                   output_sample_rate: int = 24000) -> bytes:
    """
    Upsample audio using scipy's polyphase resampling.
    16kHz to 24kHz (ratio 3:2); needed to send audio to Realtime API
    """
    try:
        # Convert input to numpy array if needed
        if isinstance(input_buffer, bytes):
            input_array = np.frombuffer(input_buffer, dtype=np.int16)
        else:
            input_array = input_buffer

        # Resample using polyphase filter (3:2 ratio for 16kHz to 24kHz)
        resampled_float = resample_poly(input_array.astype(np.float32), 3, 2)

        # Convert back to int16 with proper clipping
        output_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return output_int16.tobytes()

    except Exception as e:
        logger.error(f'Error during upsampling rate conversion: {e}')
        raise


# Constant for silence payload (matching Node.js hexSilencePayload)
SILENCE_PAYLOAD = np.array([0xff, 0xf8] * 320, dtype=np.uint8).tobytes()

# Record all audio configuration to validate
RECORD_ALL_AUDIO = os.getenv('RECORD_ALL_AUDIO', 'false').lower() == 'true'
# Event logging configuration / OpenAI realtime events
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'response.audio.delta'
]


# handle websocket state for call management
class WebSocketState:
    def __init__(self):
        self.is_active = True
        self.vonage_ws_active = True
        self.openai_ws_active = True
        self._shutting_down = False  # Track intentional shutdown

    def close(self):
        """Initiate graceful shutdown"""
        if not self._shutting_down:
            self._shutting_down = True
            self.is_active = False
            # Don't immediately mark websockets as inactive
            # Let them close properly in their own loops

    @property
    def should_continue(self):
        """Check if we should continue processing"""
        return self.is_active and not self._shutting_down

    def mark_vonage_inactive(self):
        """Mark Vonage websocket as inactive"""
        self.vonage_ws_active = False
        if not self.openai_ws_active:
            self.is_active = False

    def mark_openai_inactive(self):
        """Mark OpenAI websocket as inactive"""
        self.openai_ws_active = False
        if not self.vonage_ws_active:
            self.is_active = False


websocket_state = WebSocketState()


# add a recording handler to validate audio
from recording_manager import RecordingManager

# Initialize global recording manager
recording_manager = RecordingManager()



async def send_initial_greeting(openai_ws):
    """Send initial conversation item with proper audio output request"""
    logger.info("Sending initial greeting...")

    # Create initial message
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there, I am Chester Bennington, your AI medical assistant. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    logger.info("Sent initial conversation item")

    # Request response with explicit audio output settings
    response_request = {
        "type": "response.create",
        "options": {
            "output_audio": True,
            "output_format": "pcm16"
        }
    }
    await openai_ws.send(json.dumps(response_request))
    logger.info("Sent response request with audio output enabled")


async def initialize_session(openai_ws):
    """Initializes the OpenAI session with specific parameters"""
    logger.info("Initializing session with OpenAI...")

    # Send session settings
    await openai_ws.send(json.dumps(OPENAI_SESSION_SETTINGS))
    logger.info("Sent session settings")

    # Send initial greeting
    initial_conversation = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Greet the user with 'Hello there, I am Chester, your AI medical assistant. How can I help you?'"
            }]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation))

    # Create response
    await openai_ws.send(json.dumps({"type": "response.create"}))
    logger.info("Initialization complete")
async def handle_speech_started_event(openai_ws, last_assistant_item, latest_media_timestamp,
                                      response_start_timestamp_vonage, openai_bytes):
    """Handles interruption when caller starts speaking"""
    logger.info("Handle interruption when the caller's speech starts...")
    try:
        if isinstance(latest_media_timestamp, str) and isinstance(response_start_timestamp_vonage, str):
            latest_media_dt = datetime.strptime(latest_media_timestamp, "%Y-%m-%d %H:%M:%S")
            response_start_dt = datetime.strptime(response_start_timestamp_vonage, "%Y-%m-%d %H:%M:%S")
            elapsed_time = latest_media_dt - response_start_dt

            logger.info(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time} ms")
            truncate_event = {
                "type": "conversation.item.truncate",
                "item_id": last_assistant_item,
                "content_index": 0,
                "audio_end_ms": str(elapsed_time)
            }
            await openai_ws.send(json.dumps(truncate_event))

        # Clear buffer
        openai_bytes.clear()

        # Create new response to get OpenAI talking again
        new_response = {
            "type": "response.create"
        }
        await openai_ws.send(json.dumps(new_response))
        logger.info("Created new response after interruption")

    except Exception as e:
        logger.error(f"Error handling speech interruption: {e}")
        # Still clear buffer and create new response even if timing fails
        openai_bytes.clear()
        await openai_ws.send(json.dumps({"type": "response.create"}))

    return None, None


@app.websocket("/ws")
async def handle_websocket(websocket: WebSocket):
    """Main WebSocket handler with improved cleanup"""
    logger.info("Handle WebSocket connections between Vonage and OpenAI...")
    await websocket.accept()

    # Connection state
    latest_media_timestamp = 0
    last_assistant_item = None
    response_start_timestamp_vonage = None

    # Audio buffer management (matching Node.js)
    openai_bytes = bytearray()
    stream_to_vg_index = 0
    BUFFER_SIZE = 640  # 640-byte packet for linear16 / 16 kHz
    SILENCE_PAYLOAD = np.array([0xff, 0xf8] * 320, dtype=np.uint8).tobytes()
    STREAM_TIMER_MS = 18

    # Tasks to be cancelled on cleanup
    tasks = []
    openai_ws = None

    logger.info('Opening WebSocket connection to OpenAI Realtime')
    try:
        async with websockets.connect(
                OPENAI_WS_URL,
                additional_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
        ) as openai_ws:
            # Initialize the session first
            await initialize_session(openai_ws)

            async def stream_timer():
                """Handle streaming audio to Vonage with precise timing"""
                try:
                    while websocket_state.vonage_ws_active:
                        await asyncio.sleep(STREAM_TIMER_MS / 1000)  # Convert ms to seconds

                        if len(openai_bytes) > 0:
                            nonlocal stream_to_vg_index
                            stream_packet = bytes(openai_bytes[
                                                  stream_to_vg_index:stream_to_vg_index + BUFFER_SIZE
                                                  ])
                            stream_to_vg_index += BUFFER_SIZE

                            if len(stream_packet) == BUFFER_SIZE and websocket_state.vonage_ws_active:
                                await websocket.send_bytes(stream_packet)
                            else:
                                # Prevent index from increasing forever
                                stream_to_vg_index -= BUFFER_SIZE
                                if websocket_state.vonage_ws_active:
                                    await websocket.send_bytes(SILENCE_PAYLOAD)
                        else:
                            if websocket_state.vonage_ws_active:
                                await websocket.send_bytes(SILENCE_PAYLOAD)
                except Exception as e:
                    logger.error(f"Error in stream_timer: {e}")
                    websocket_state.vonage_ws_active = False

            # Create tasks with improved error handling
            tasks = [
                asyncio.create_task(receive_from_vonage(websocket, openai_ws, latest_media_timestamp)),
                asyncio.create_task(send_to_vonage(openai_ws, websocket, last_assistant_item,
                                                   response_start_timestamp_vonage, openai_bytes,
                                                   latest_media_timestamp)),
                asyncio.create_task(stream_timer())
            ]

            # Wait for tasks to complete or websocket_state to close
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled due to connection cleanup")
            except Exception as e:
                logger.error(f"Error in tasks: {e}")

    except Exception as e:
        logger.error(f"Error in handle_websocket: {e}")
    finally:
        logger.info("Initiating cleanup of all connections")
        websocket_state.close()

        # Cancel all running tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")

        # Close OpenAI websocket if it exists
        try:
            if openai_ws is not None:
                await openai_ws.close()
                logger.info("OpenAI websocket closed successfully")
        except Exception as e:
            logger.error(f"Error closing OpenAI websocket: {e}")

        # Close Vonage websocket
        try:
            await websocket.close()
            logger.info("Vonage websocket closed successfully")
        except Exception as e:
            logger.error(f"Error closing Vonage websocket: {e}")


async def receive_from_vonage(websocket: WebSocket, openai_ws, latest_media_timestamp):
    """Handles receiving audio from Vonage and sending to OpenAI"""
    logger.info("...Vonage --to-- Open AI...")
    try:
        while websocket_state.is_active and websocket_state.vonage_ws_active:
            try:
                message = await websocket.receive()

                # Check for specific websocket disconnect message
                if isinstance(message, dict) and message.get('type') == 'websocket.disconnect':
                    logger.info("Received websocket disconnect message")
                    websocket_state.vonage_ws_active = False
                    break

                if message is None:
                    await asyncio.sleep(0.020)  # Add small delay on empty messages
                    continue

                # Handle text messages (like connection events)
                if 'text' in message:
                    try:
                        data = json.loads(message['text'])
                        if data['event'] == 'websocket:connected':
                            logger.info("WebSocket Connected.")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse text message")
                        continue

                # Handle binary audio data
                if 'bytes' in message and websocket_state.openai_ws_active:
                    now = datetime.now()
                    latest_media_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                    if RECORD_ALL_AUDIO:
                        try:
                            # Record raw audio from Vonage before processing
                            audio_from_vg_filename = f'./recordings/{websocket.headers.get("peer_uuid", "unknown")}_rec_from_vg.raw'
                            with open(audio_from_vg_filename, 'ab') as f:
                                f.write(message['bytes'])
                        except Exception as e:
                            logger.error(f"Error writing to file {audio_from_vg_filename}: {e}")

                    # Upsample from 16kHz to 24kHz for OpenAI
                    processed_audio = upsample_audio(message['bytes'], 16000, 24000)

                    if RECORD_ALL_AUDIO:
                        try:
                            # Record processed audio before sending to OpenAI
                            audio_to_oai_filename = f'./recordings/{websocket.headers.get("peer_uuid", "unknown")}_rec_to_oai.raw'
                            with open(audio_to_oai_filename, 'ab') as f:
                                f.write(processed_audio)
                        except Exception as e:
                            logger.error(f"Error writing to file {audio_to_oai_filename}: {e}")

                    if websocket_state.openai_ws_active:
                        # Prepare and send audio to OpenAI
                        payload_to_openai = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(processed_audio).decode('utf-8')
                        }
                        try:
                            await openai_ws.send(json.dumps(payload_to_openai))
                            logger.info("Sent audio to OpenAI")
                        except Exception as e:
                            logger.error(f"Error sending audio to OpenAI: {e}")
                            if "not connected" in str(e):
                                websocket_state.openai_ws_active = False
                            continue

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                websocket_state.vonage_ws_active = False
                break
            except Exception as e:
                if "disconnect message" in str(e):
                    logger.info("WebSocket disconnected gracefully")
                    websocket_state.vonage_ws_active = False
                    break
                else:
                    logger.error(f"Error in receive loop: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error in receive_from_vonage: {e}")
        if not str(e).startswith("Cannot call"):  # Ignore normal disconnect errors
            websocket_state.vonage_ws_active = False
    finally:
        logger.info("Exiting receive_from_vonage")


async def send_to_vonage(openai_ws, websocket, last_assistant_item,
                         response_start_timestamp_vonage, openai_bytes,
                         latest_media_timestamp):
    """Handles receiving from OpenAI and preparing audio for Vonage"""
    logger.info("...Open AI --to-- Vonage...")

    # Get peer UUID for recordings
    peer_uuid = websocket.headers.get("peer_uuid", "unknown")
    recording_files = await recording_manager.initialize_recording_files(peer_uuid)

    try:
        async for openai_message in openai_ws:
            if not websocket_state.is_active:
                break

            response = json.loads(openai_message)
            logger.info(f"OpenAI message type: {response.get('type')}")

            if response.get('type') == 'response.audio.delta':
                if response.get('delta'):
                    # Decode base64 audio from OpenAI
                    payload_in_from_oai = base64.b64decode(response['delta'])

                    # Record raw audio from OpenAI if recording enabled
                    if recording_manager.record_all_audio and recording_files:
                        await recording_manager.write_audio(
                            recording_files['from_oai'],
                            payload_in_from_oai
                        )

                    # Downsample from 24kHz to 16kHz for Vonage
                    payload_to_vg = downsample_audio(payload_in_from_oai, 24000, 16000)

                    # Record downsampled audio if recording enabled
                    if recording_manager.record_all_audio and recording_files:
                        await recording_manager.write_audio(
                            recording_files['to_vg_1'],
                            payload_to_vg
                        )

                    if websocket_state.vonage_ws_active:
                        openai_bytes.extend(payload_to_vg)

                    if response_start_timestamp_vonage is None:
                        response_start_timestamp_vonage = latest_media_timestamp

                    if response.get('item_id'):
                        last_assistant_item = response['item_id']

            elif response.get('type') == 'input_audio_buffer.speech_started':
                logger.info("Speech started detected.")
                openai_bytes.clear()
                await openai_ws.send(json.dumps({"type": "response.cancel"}))


            elif response.get('type') == 'response.done':

                logger.info(f"Response done: {response}")

                # Only try to access status_details if response and response.response exist

                if response.get('response') and response.get('response', {}).get('status_details'):

                    if response['response']['status_details'].get('type') == 'failed':
                        logger.error(f"OpenAI error: {response['response']['status_details'].get('error')}")

                # Don't close the connection just because we got a response.done

                continue  # Add this to keep the connection alive

    except Exception as e:
        logger.error(f"Error in send_to_vonage: {e}")
        websocket_state.close()


async def receive_from_vonage(websocket: WebSocket, openai_ws, latest_media_timestamp):
    """Handles receiving audio from Vonage and sending to OpenAI"""
    logger.info("...Vonage --to-- Open AI...")

    # Get peer UUID for recordings
    peer_uuid = websocket.headers.get("peer_uuid", "unknown")
    recording_files = await recording_manager.initialize_recording_files(peer_uuid)

    try:
        while websocket_state.is_active and websocket_state.vonage_ws_active:
            try:
                message = await websocket.receive()

                if isinstance(message, dict) and message.get('type') == 'websocket.disconnect':
                    logger.info("Received websocket disconnect message")
                    websocket_state.vonage_ws_active = False
                    break

                if message is None:
                    await asyncio.sleep(0.020)  # Add small delay on empty messages
                    continue

                if 'text' in message:
                    try:
                        data = json.loads(message['text'])
                        if data['event'] == 'websocket:connected':
                            logger.info("WebSocket Connected.")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse text message")
                        continue

                if 'bytes' in message and websocket_state.openai_ws_active:
                    now = datetime.now()
                    latest_media_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                    # Record raw audio from Vonage if recording enabled
                    if recording_manager.record_all_audio and recording_files:
                        await recording_manager.write_audio(
                            recording_files['from_vg'],
                            message['bytes']
                        )

                    # Upsample from 16kHz to 24kHz for OpenAI
                    processed_audio = upsample_audio(message['bytes'], 16000, 24000)

                    # Record processed audio if recording enabled
                    if recording_manager.record_all_audio and recording_files:
                        await recording_manager.write_audio(
                            recording_files['to_oai'],
                            processed_audio
                        )

                    if websocket_state.openai_ws_active:
                        payload_to_openai = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(processed_audio).decode('utf-8')
                        }
                        try:
                            await openai_ws.send(json.dumps(payload_to_openai))
                            logger.info("Sent audio to OpenAI")
                        except Exception as e:
                            logger.error(f"Error sending audio to OpenAI: {e}")
                            if "not connected" in str(e):
                                websocket_state.openai_ws_active = False
                            continue

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                websocket_state.vonage_ws_active = False
                break
            except Exception as e:
                if "disconnect message" in str(e):
                    logger.info("WebSocket disconnected gracefully")
                    websocket_state.vonage_ws_active = False
                    break
                else:
                    logger.error(f"Error in receive loop: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error in receive_from_vonage: {e}")
        if not str(e).startswith("Cannot call"):  # Ignore normal disconnect errors
            websocket_state.vonage_ws_active = False
    finally:
        logger.info("Exiting receive_from_vonage")


@app.get("/event")
async def handle_event(request: Request):
    """Handle Vonage event callbacks with proper cleanup"""
    params = dict(request.query_params)
    logger.info(f"Received event: {params}")

    # Check for call completion
    if params.get('status') == 'completed':
        logger.info("Call completed, initiating websocket cleanup")
        websocket_state.close()

    return "OK"


@app.get("/answer")
async def answer_call(request: Request, from_: str = Query(..., alias="from")):
    """Handle incoming call"""
    print("Query params:", dict(request.query_params))

    ws_url = f"wss://{request.base_url.hostname}/ws"
    if "ngrok" not in request.base_url.hostname:
        ws_url = f"ws://{request.base_url.hostname}:{request.base_url.port}/ws"

    return [
        {
            "action": "talk",
            "text": "Please wait while I connect you to our medical assistant, Chester Bennington",
            "language": "en-US",
            "style": 2,
            "premium": True,
            "level": 1,
            'loop': 1,
        },
        {
            "action": "connect",
            "endpoint": [
                {
                    "type": "websocket",
                    "uri": ws_url,
                    "content-type": "audio/l16;rate=16000",
                    "headers": {
                        "peer_uuid": from_
                    }
                }
            ]
        }
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)