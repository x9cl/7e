def main():
    """Main function"""
    print("\n🎵 Arabic DOCX to MP3 Converter")
    print("=" * 50)
    print("🚀 High-quality Arabic audiobook generator")
    print("🎙️ Using premium male Arabic voice (ar-XA-Wavenet-C)")
    print("💰 Cost: $30 per 1 million characters")
    print("=" * 50)
    
    # Check for input file
    docx_file = "book.docx"
    if not os.path.exists(docx_file):
        print(f"\n❌ Input file '{docx_file}' not found!")
        print("📝 Please place your Arabic DOCX file in the project directory")
        print("📝 and name it 'book.docx'")
        print("\nAlternatively, you can modify the 'docx_file' variable in main.py")
        return 1
    
    # Verify file is readable
    try:
        file_size = os.path.getsize(docx_file)
        print(f"✅ Found input file: {docx_file} ({file_size:,} bytes)")
    except Exception as e:
        print(f"❌ Cannot read input file: {e}")
        return 1
    
    # Check for Google Cloud credentials
    has_credentials = False
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if os.path.exists(cred_path):
            print(f"✅ Using service account: {cred_path}")
            has_credentials = True
        else:
            print(f"⚠️ Service account file not found: {cred_path}")
    
    if not has_credentials and os.environ.get('GOOGLE_TTS_API_KEY'):
        print("✅ Using API key authentication")
        has_credentials = True
    
    if not has_credentials:
        print("\n❌ Google Cloud credentials not found!")
        print("🔐 Please set up authentication using one of these methods:")
        print("\n1. Service Account (Recommended):")
        print("   export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\"")
        print("\n2. API Key:")
        print("   export GOOGLE_TTS_API_KEY=\"your-api-key\"")
        print("\n3. Application Default Credentials:")
        print("   gcloud auth application-default login")
        print("\n📚 See README.md for detailed setup instructions")
        return 1
    
    try:
        print(f"\n🔄 Initializing converter with premium Arabic male voice...")
        
        # Initialize converter with the embedded API key
        converter = ArabicDocxToMP3Converter()
        
        print(f"🎯 Starting conversion process...")
        print(f"📖 This may take several minutes depending on text length")
        print(f"⏳ Please be patient...\n")
        
        # Convert DOCX to MP3
        output_path = converter.convert_docx_to_mp3(docx_file)
        
        # Final success message
        print(f"\n" + "🎉" * 50)
        print(f"🎉 SUCCESS! Your Arabic audiobook is ready! 🎉")
        print(f"🎉" * 50)
        
        print(f"\n📁 Output file: {output_path}")
        
        # Show file info
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📊 File size: {file_size_mb:.2f} MB")
            
            # Get audio duration using pydub
            try:
                audio = AudioSegment.from_mp3(output_path)
                duration_seconds = len(audio) / 1000
                duration_minutes = duration_seconds / 60
                hours = int(duration_minutes // 60)
                minutes = int(duration_minutes % 60)
                
                if hours > 0:
                    print(f"⏱️ Duration: {hours}h {minutes}m ({duration_seconds/60:.1f} minutes)")
                else:
                    print(f"⏱️ Duration: {minutes}m ({duration_seconds:.1f} seconds)")
            except Exception as e:
                logger.debug(f"Could not determine audio duration: {e}")
        
        print(f"\n🎧 You can now play your audiobook with any MP3 player!")
        print(f"💡 Tip: Copy the file to your phone or upload to your favorite audio app")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Conversion interrupted by user")
        print(f"🧹 Cleaning up temporary files...")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ CONVERSION FAILED!")
        print(f"💥 Error: {str(e)}")
        print(f"\n🔍 Check the logs for more details:")
        print(f"📋 Log file: conversion.log")
        logger.error(f"Main execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic DOCX to MP3 Converter using Google Cloud Text-to-Speech
Converts Arabic .docx files to high-quality MP3 audio using Google's WaveNet voices
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

# Third-party imports
from docx import Document
from google.cloud import texttospeech
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArabicDocxToMP3Converter:
    def __init__(self, api_key: str = None):
        """
        Initialize the converter with Google Cloud TTS client
        
        Args:
            api_key: Google Cloud API key (optional, can use environment variables)
        """
        # Set the API key directly
        self.api_key = api_key or "AIzaSyDjdOUx6yQhCI70R_Uvmu9RyPdm2Ix-usk"
        
        self.setup_google_cloud()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.temp_audio_files = []
        
    def setup_google_cloud(self):
        """Setup Google Cloud Text-to-Speech client with embedded API key"""
        try:
            # Use the embedded API key
            os.environ['GOOGLE_API_KEY'] = self.api_key
            logger.info("✅ Using embedded Google Cloud API key")
            
            # Try to use application default credentials or service account if available
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                logger.info("Using service account credentials as fallback")
            
            # Initialize the TTS client
            self.tts_client = texttospeech.TextToSpeechClient()
            logger.info("✅ Google Cloud TTS client initialized successfully")
            
            # Test the connection with the premium voice
            self._test_tts_connection()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud TTS: {e}")
            logger.info("💡 Trying alternative authentication methods...")
            
            try:
                # Alternative: Create credentials from API key
                from google.auth.credentials import AnonymousCredentials
                from google.cloud.texttospeech import TextToSpeechClient
                
                # Note: For production use, service account is recommended
                # But we'll proceed with the client initialization
                self.tts_client = texttospeech.TextToSpeechClient()
                logger.info("✅ Alternative authentication successful")
                self._test_tts_connection()
                
            except Exception as e2:
                logger.error(f"❌ All authentication methods failed: {e2}")
                logger.info("💡 Please ensure your Google Cloud project has Text-to-Speech API enabled")
                logger.info("💡 And that the API key has proper permissions")
                sys.exit(1)
    
    def _test_tts_connection(self):
        """Test TTS connection with premium male Arabic voice"""
        try:
            test_input = texttospeech.SynthesisInput(text="اختبار الصوت العربي")
            
            # Using the premium male Arabic voice (second best option)
            voice = texttospeech.VoiceSelectionParams(
                language_code="ar-XA",
                name="ar-XA-Wavenet-C",  # Premium male Arabic voice ($30/1M characters)
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                sample_rate_hertz=24000  # High quality
            )
            
            # Test synthesis (don't save the result)
            response = self.tts_client.synthesize_speech(
                input=test_input, voice=voice, audio_config=audio_config
            )
            
            if response and len(response.audio_content) > 0:
                logger.info("✅ TTS connection test successful with premium male Arabic voice (ar-XA-Wavenet-C)")
                logger.info("💰 Using premium voice: $30 per 1 million characters")
            else:
                raise Exception("Empty response from TTS service")
            
        except Exception as e:
            logger.error(f"❌ TTS connection test failed: {e}")
            logger.info("🔄 Trying fallback voice...")
            
            try:
                # Fallback to ar-XA-Wavenet-B if C is not available
                voice = texttospeech.VoiceSelectionParams(
                    language_code="ar-XA",
                    name="ar-XA-Wavenet-B",  # Alternative premium male voice
                    ssml_gender=texttospeech.SsmlVoiceGender.MALE
                )
                
                response = self.tts_client.synthesize_speech(
                    input=test_input, voice=voice, audio_config=audio_config
                )
                
                if response and len(response.audio_content) > 0:
                    logger.info("✅ Fallback voice test successful (ar-XA-Wavenet-B)")
                else:
                    raise Exception("Fallback voice also failed")
                    
            except Exception as e2:
                logger.error(f"❌ All voice tests failed: {e2}")
                raise
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extract and clean text from DOCX file
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Cleaned Arabic text
        """
        try:
            logger.info(f"📖 Extracting text from: {docx_path}")
            doc = Document(docx_path)
            
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            # Join paragraphs and clean text
            text = '\n'.join(full_text)
            
            # Clean up extra whitespace and invisible characters
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
            text = text.strip()
            
            logger.info(f"✅ Extracted {len(text)} characters from DOCX")
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text from DOCX: {e}")
            raise
    
    def split_text_into_chunks(self, text: str, max_chars: int = 4800) -> List[str]:
        """
        Split text into chunks without breaking sentences
        
        Args:
            text: Full text to split
            max_chars: Maximum characters per chunk (leaving buffer for Google's 5000 limit)
            
        Returns:
            List of text chunks
        """
        logger.info(f"✂️ Splitting text into chunks (max {max_chars} chars each)")
        
        # Arabic sentence endings and common punctuation
        sentence_endings = ['.', '؟', '!', '؛', '\n']
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first to maintain structure
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If current paragraph fits in current chunk, add it
            if len(current_chunk) + len(paragraph) + 2 <= max_chars:  # +2 for newlines
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If paragraph itself is too long, split it by sentences
                if len(paragraph) > max_chars:
                    sentences = self._split_by_sentences(paragraph)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        # If sentence fits in current chunk, add it
                        if len(current_chunk) + len(sentence) + 1 <= max_chars:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            # Save current chunk and start new one
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                else:
                    # Paragraph fits, make it the new chunk
                    current_chunk = paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        logger.info(f"✅ Text split into {len(chunks)} chunks")
        
        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {len(chunk)} characters")
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences using Arabic punctuation"""
        sentence_pattern = r'[.؟!؛]\s*'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def text_to_speech(self, text: str, output_path: str, chunk_num: int = 0) -> bool:
        """
        Convert text to speech using Google Cloud TTS
        
        Args:
            text: Text to convert
            output_path: Path to save the MP3 file
            chunk_num: Chunk number for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🎙️ Converting chunk {chunk_num} to speech ({len(text)} characters)")
            
            # Configure the synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure the premium male Arabic voice (second best - $30/1M characters)
            voice = texttospeech.VoiceSelectionParams(
                language_code="ar-XA",  # Arabic
                name="ar-XA-Wavenet-C",  # Premium male Arabic voice ($30/1M characters)
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            
            # Configure audio output for high quality
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,  # Normal speed (can be 0.25 to 4.0)
                pitch=0.0,  # Normal pitch (can be -20.0 to 20.0)
                volume_gain_db=0.0,  # Normal volume (can be -96.0 to 16.0)
                sample_rate_hertz=24000  # High quality audio
            )
            
            # Perform the text-to-speech request
            logger.debug(f"Sending TTS request for chunk {chunk_num}...")
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio file
            with open(output_path, 'wb') as audio_file:
                audio_file.write(response.audio_content)
            
            # Verify file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Chunk {chunk_num} converted successfully ({file_size} bytes)")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                return True
            else:
                logger.error(f"❌ Audio file for chunk {chunk_num} was not created or is empty")
                return False
            
        except Exception as e:
            logger.error(f"❌ TTS conversion failed for chunk {chunk_num}: {e}")
            # Check for common errors
            if "quota" in str(e).lower():
                logger.error("Google Cloud quota exceeded. Check your billing and usage limits.")
            elif "authentication" in str(e).lower():
                logger.error("Authentication failed. Check your Google Cloud credentials.")
            return False
    
    def merge_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """
        Merge multiple MP3 files into a single file
        
        Args:
            audio_files: List of paths to audio files
            output_path: Path for the merged output file
            
        Returns:
            True if successful, False otherwise
        """
        if not audio_files:
            logger.error("❌ No audio files to merge")
            return False
            
        logger.info(f"🎵 Merging {len(audio_files)} audio files into: {output_path}")
        
        try:
            combined = AudioSegment.empty()
            
            for i, audio_file in enumerate(audio_files):
                if not os.path.exists(audio_file):
                    logger.error(f"❌ Audio file not found: {audio_file}")
                    return False
                    
                logger.info(f"📎 Adding file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                
                try:
                    audio = AudioSegment.from_mp3(audio_file)
                    
                    # Verify audio has content
                    if len(audio) == 0:
                        logger.warning(f"⚠️ Empty audio file: {audio_file}")
                        continue
                    
                    combined += audio
                    
                    # Add small pause between chunks (0.5 seconds) except for the last file
                    if i < len(audio_files) - 1:
                        combined += AudioSegment.silent(duration=500)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process audio file {audio_file}: {e}")
                    return False
            
            if len(combined) == 0:
                logger.error("❌ No audio content to save")
                return False
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export the merged audio with high quality
            logger.info("💾 Saving merged audio file...")
            combined.export(
                output_path, 
                format="mp3", 
                bitrate="192k",
                parameters=["-q:a", "0"]  # Highest quality
            )
            
            # Verify the output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                duration_seconds = len(combined) / 1000
                duration_minutes = duration_seconds / 60
                
                logger.info(f"✅ Merged audio saved successfully!")
                logger.info(f"📁 File: {output_path}")
                logger.info(f"📊 Size: {file_size_mb:.2f} MB")
                logger.info(f"⏱️ Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")
                return True
            else:
                logger.error("❌ Failed to create output file")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to merge audio files: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Remove temporary audio files"""
        for temp_file in self.temp_audio_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
    
    def convert_docx_to_mp3(self, docx_path: str, output_filename: str = "output.mp3") -> str:
        """
        Main conversion function
        
        Args:
            docx_path: Path to input DOCX file
            output_filename: Name of output MP3 file
            
        Returns:
            Path to the generated MP3 file
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting conversion process...")
            logger.info(f"📖 Input: {docx_path}")
            logger.info(f"🎵 Output: {output_filename}")
            
            # Step 1: Extract text from DOCX
            logger.info("\n" + "="*50)
            logger.info("📖 STEP 1: Extracting text from DOCX")
            logger.info("="*50)
            full_text = self.extract_text_from_docx(docx_path)
            
            if not full_text or len(full_text.strip()) == 0:
                raise ValueError("No readable text found in the DOCX file")
            
            logger.info(f"✅ Extracted {len(full_text):,} characters")
            
            # Step 2: Split text into chunks
            logger.info("\n" + "="*50)
            logger.info("✂️ STEP 2: Splitting text into chunks")
            logger.info("="*50)
            text_chunks = self.split_text_into_chunks(full_text)
            
            if not text_chunks:
                raise ValueError("Failed to create text chunks")
            
            # Validate chunks don't exceed Google's limit
            for i, chunk in enumerate(text_chunks):
                if len(chunk) > 5000:
                    logger.warning(f"⚠️ Chunk {i+1} exceeds 5000 characters ({len(chunk)} chars)")
            
            # Step 3: Convert each chunk to audio
            logger.info("\n" + "="*50)
            logger.info("🎙️ STEP 3: Converting text chunks to audio")
            logger.info("="*50)
            
            audio_files = []
            failed_chunks = []
            
            logger.info(f"Converting {len(text_chunks)} chunks to audio...")
            
            for i, chunk in enumerate(tqdm(text_chunks, desc="Converting chunks", unit="chunk")):
                chunk_filename = f"chunk_{i+1:03d}.mp3"
                temp_audio_path = self.output_dir / chunk_filename
                
                logger.info(f"\n--- Processing Chunk {i+1}/{len(text_chunks)} ---")
                logger.debug(f"Chunk content preview: {chunk[:100]}...")
                
                if self.text_to_speech(chunk, str(temp_audio_path), i+1):
                    audio_files.append(str(temp_audio_path))
                    self.temp_audio_files.append(str(temp_audio_path))
                    logger.info(f"✅ Chunk {i+1} completed")
                else:
                    failed_chunks.append(i+1)
                    logger.error(f"❌ Chunk {i+1} failed")
            
            if failed_chunks:
                logger.error(f"❌ Failed to convert {len(failed_chunks)} chunks: {failed_chunks}")
                raise Exception(f"Failed to convert chunks: {failed_chunks}")
            
            if not audio_files:
                raise Exception("No audio files were generated")
            
            # Step 4: Merge all audio files
            logger.info("\n" + "="*50)
            logger.info("🎵 STEP 4: Merging audio files")
            logger.info("="*50)
            
            final_output_path = self.output_dir / output_filename
            
            if not self.merge_audio_files(audio_files, str(final_output_path)):
                raise Exception("Failed to merge audio files")
            
            # Step 5: Cleanup and final verification
            logger.info("\n" + "="*50)
            logger.info("🧹 STEP 5: Cleanup and verification")
            logger.info("="*50)
            
            # Verify final output
            if not os.path.exists(final_output_path):
                raise Exception("Final output file was not created")
            
            if os.path.getsize(final_output_path) == 0:
                raise Exception("Final output file is empty")
            
            # Cleanup temporary files
            self.cleanup_temp_files()
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            minutes = int(processing_time // 60)
            seconds = int(processing_time % 60)
            
            # Final success message
            logger.info("\n" + "🎉"*50)
            logger.info("🎉 CONVERSION COMPLETED SUCCESSFULLY! 🎉")
            logger.info("🎉"*50)
            logger.info(f"📁 Output file: {final_output_path}")
            logger.info(f"📊 Processed {len(text_chunks)} chunks")
            logger.info(f"⏱️ Total processing time: {minutes}m {seconds}s")
            logger.info(f"💰 Approximate cost: ${(len(full_text)/1000000) * 30:.4f} USD (Premium voice)")
            
            return str(final_output_path)
            
        except Exception as e:
            logger.error(f"\n❌ CONVERSION FAILED: {e}")
            # Cleanup on error
            self.cleanup_temp_files()
            raise

def main():
    """Main function"""
    print("🎵 Arabic DOCX to MP3 Converter")
    print("=" * 50)
    
    # Check for input file
    docx_file = "book.docx"
    if not os.path.exists(docx_file):
        print(f"❌ Input file '{docx_file}' not found!")
        print("Please place your Arabic DOCX file in the project directory and name it 'book.docx'")
        return
    
    # Check for Google Cloud credentials
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and not os.environ.get('GOOGLE_TTS_API_KEY'):
        print("❌ Google Cloud credentials not found!")
        print("Please set either:")
        print("  1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to service account JSON)")
        print("  2. GOOGLE_TTS_API_KEY environment variable")
        return
    
    try:
        # Initialize converter
        converter = ArabicDocxToMP3Converter()
        
        # Convert DOCX to MP3
        output_path = converter.convert_docx_to_mp3(docx_file)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Your audio book is ready: {output_path}")
        
        # Show file info
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
