# content_analysis_service.py - Simplified Version
import os
import sys
import json
import time
import logging
import subprocess
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the src directory to Python path
current_dir = Path(__file__).parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

# Load environment variables
load_dotenv()

# Configure APIs
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Define paths
BACKEND_DIR = Path(__file__).parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"
AUDIO_DIR = BACKEND_DIR / "audio"
ANALYTICS_DIR = BACKEND_DIR / "analytics"
CONTENT_ANALYSIS_FILE = ANALYTICS_DIR / "content_analysis.json"

# Create directories if they don't exist
AUDIO_DIR.mkdir(exist_ok=True)
ANALYTICS_DIR.mkdir(exist_ok=True)

async def wait_for_valid_video(video_path, max_attempts=10, delay=2):
    """Wait for the video to be fully written and valid"""
    for attempt in range(max_attempts):
        try:
            # Try to get video info using FFprobe
            cmd = ["ffprobe", "-v", "error", str(video_path)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            
            # If there's no error, the video is valid
            if process.returncode == 0:
                return True
                
            # If there's a "moov atom not found" error, the video is still being written
            if b"moov atom not found" in stderr:
                logging.info(f"Video still being written (attempt {attempt+1}/{max_attempts})")
                await asyncio.sleep(delay)
                continue
                
            # Other errors
            logging.error(f"Error validating video: {stderr.decode()}")
            
        except Exception as e:
            logging.error(f"Exception validating video: {str(e)}")
            
        await asyncio.sleep(delay)
    
    return False

async def extract_audio(video_path, output_path):
    """Extract audio from video file using FFmpeg"""
    try:
        # Use a more flexible FFmpeg command
        command = [
            "ffmpeg", 
            "-i", str(video_path),
            "-vn",  # Skip video
            "-acodec", "libmp3lame",  # Use MP3 codec
            "-q:a", "2",  # Quality setting
            "-y",  # Overwrite output
            str(output_path)
        ]
        
        # Run the command
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            stderr_text = stderr.decode()
            logging.error(f"FFmpeg error: {stderr_text}")
            
            # Check if the error is about no audio stream
            if "Stream map 'a' matches no streams" in stderr_text or "does not contain any stream" in stderr_text:
                logging.error("Video has no audio stream")
                return False, "no_audio"
            return False, stderr_text
            
        return True, None
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        return False, str(e)

async def create_dummy_audio(output_path):
    """Create a dummy audio file with silence"""
    try:
        dummy_command = [
            "ffmpeg", 
            "-f", "lavfi", 
            "-i", "anullsrc=r=44100:cl=mono", 
            "-t", "10",  # 10 seconds of silence
            "-q:a", "2",
            "-y",
            str(output_path)
        ]
        process = subprocess.Popen(
            dummy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"Error creating dummy audio: {stderr.decode()}")
            return False
        return True
    except Exception as e:
        logging.error(f"Exception creating dummy audio: {str(e)}")
        return False

async def transcribe_audio(audio_path):
    """Transcribe audio file using Deepgram API with timestamps"""
    try:
        # Check if API key is configured
        if not DEEPGRAM_API_KEY:
            logging.error("Deepgram API key not configured")
            return None
            
        # Check if file exists
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return None
            
        # Get file size
        file_size = os.path.getsize(audio_path)
        logging.info(f"Audio file size: {file_size} bytes")
        
        # Read the audio file
        with open(audio_path, "rb") as audio:
            audio_data = audio.read()
        
        # Prepare the API request
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "punctuate": "true",
            "paragraphs": "true",
            "summarize": "v2",
            "diarize": "false",
            "model": "nova"
        }
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/mpeg"
        }
        
        logging.info(f"Sending request to Deepgram API: {url}")
        
        # Send the request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, 
                                   params=params, 
                                   headers=headers, 
                                   data=audio_data,
                                   timeout=aiohttp.ClientTimeout(total=60)) as response:
                
                status = response.status
                logging.info(f"Deepgram API response status: {status}")
                
                response_text = await response.text()
                
                if status != 200:
                    logging.error(f"Deepgram API error ({status}): {response_text}")
                    return None
                
                try:
                    result = json.loads(response_text)
                    logging.info("Successfully parsed Deepgram response")
                    
                    # Remove the words array to reduce response size
                    if 'results' in result and 'channels' in result['results']:
                        for channel in result['results']['channels']:
                            if 'alternatives' in channel:
                                for alternative in channel['alternatives']:
                                    if 'words' in alternative:
                                        del alternative['words']
                    
                    return result
                    
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse Deepgram response as JSON: {response_text[:100]}...")
                    return None
                
    except aiohttp.ClientError as e:
        logging.error(f"Deepgram API request error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error transcribing audio: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    
async def process_latest_video():
    """Process the latest video to extract and analyze audio content"""
    try:
        # Find the latest uploaded video (original with audio) instead of processed
        video_files = list(UPLOAD_DIR.glob("*.*"))
        if not video_files:
            logging.info("No uploaded videos found")
            return False
            
        # Use the latest uploaded video
        video_path = video_files[0]
        
        # Define the audio output path
        audio_filename = "latest_audio.mp3"
        audio_path = AUDIO_DIR / audio_filename
        
        # Clear the audio directory
        for old_file in AUDIO_DIR.glob("*.*"):
            old_file.unlink()
        
        logging.info(f"Processing video for audio: {video_path}")
        
        # Wait for the video to be valid
        valid = await wait_for_valid_video(video_path)
        if not valid:
            logging.error("Video file is not valid after multiple attempts")
            return False
        
        # Step 1: Extract audio
        logging.info("Extracting audio...")
        audio_extracted, error = await extract_audio(video_path, audio_path)
        
        # Create a dummy result if the video has no audio
        if not audio_extracted and error == "no_audio":
            logging.info("Creating dummy analysis since video has no audio")
            
            content_analysis = {
                "timestamp": time.time(),
                "video": str(video_path),
                "has_audio": False,
                "message": "Video does not contain an audio track"
            }
            
            with open(CONTENT_ANALYSIS_FILE, "w") as f:
                json.dump(content_analysis, f, indent=2)
            
            logging.info(f"Content analysis completed with dummy data for video without audio")
            return True
                
        elif not audio_extracted:
            logging.error("Failed to extract audio")
            return False
        
        # Step 2: Transcribe audio
        logging.info("Transcribing audio...")
        transcription_result = await transcribe_audio(audio_path)
        
        if not transcription_result:
            logging.error("Failed to get transcription result")
            return False
        
        # Step 3: Save results (without duplicating emotion data)
        content_analysis = {
            "timestamp": time.time(),
            "video": str(video_path),
            "audio": str(audio_path),
            "has_audio": True,
            "transcription": transcription_result  # Store the complete result
        }
        
        with open(CONTENT_ANALYSIS_FILE, "w") as f:
            json.dump(content_analysis, f, indent=2)
        
        logging.info(f"Content analysis completed: {CONTENT_ANALYSIS_FILE}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

async def main():
    """Main function to monitor for new videos and process them"""
    logging.info("Content Analysis Service started")
    
    # Set up file monitoring
    last_modified = None
    last_processed_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Don't process more than once every 30 seconds
            if current_time - last_processed_time < 30:
                await asyncio.sleep(5)
                continue
            
            # Check for new files
            if list(UPLOAD_DIR.glob("*.*")):
                current_modified = max(f.stat().st_mtime for f in UPLOAD_DIR.glob("*.*"))
                
                # If a new file is detected or first run
                if last_modified is None or current_modified > last_modified:
                    logging.info("New video detected, waiting 3 seconds for it to be fully written...")
                    
                    # Wait for 10 seconds to ensure the file is fully written
                    await asyncio.sleep(3)
                    
                    success = await process_latest_video()
                    if success:
                        last_modified = current_modified
                        last_processed_time = time.time()
            
            # Sleep before checking again
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        logging.info("Service stopped by user")
    except Exception as e:
        logging.error(f"Service error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())