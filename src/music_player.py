"""
Music player module for emotion-based music playback.
Plays music based on detected emotions from the camera.
"""

import os
import random
import threading
import time
import pygame
from pathlib import Path
from typing import Optional, List, Dict


class EmotionMusicPlayer:
    """
    Music player that plays music based on detected emotions.
    """
    
    # Emotion to music folder mapping
    EMOTION_MAPPING = {
        'Angry': 'angry',
        'Disgust': 'disgust',
        'Fear': 'fear',
        'Happy': 'happy',
        'Sad': 'sad',
        'Surprise': 'surprise',
        'Neutral': 'neutral'
    }
    
    # Supported audio formats
    AUDIO_FORMATS = ('.mp3', '.wav', '.ogg', '.m4a', '.flac')
    
    def __init__(self, music_dir='music', volume=0.7, transition_delay=3.0):
        """
        Initialize the music player.
        
        Args:
            music_dir: Base directory containing emotion-specific music folders
            volume: Initial volume (0.0 to 1.0)
            transition_delay: Seconds to wait before switching music on emotion change
        """
        self.music_dir = Path(music_dir)
        self.volume = volume
        self.transition_delay = transition_delay
        
        # Initialize pygame (required for event handling)
        # Only initialize if not already initialized to avoid conflicts
        if not pygame.get_init():
            pygame.init()
        
        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.music.set_volume(volume)
        
        # Set up end-of-song event
        self.music_end_event = pygame.USEREVENT + 1
        pygame.mixer.music.set_endevent(self.music_end_event)
        
        # State tracking
        self.current_emotion = None
        self.current_track = None
        self.playlists: Dict[str, List[str]] = {}
        self.is_playing = False
        self.is_paused = False
        self.auto_play_next = True  # Auto-play next song when current ends
        
        # Threading for smooth transitions
        self.transition_thread = None
        self.stop_transition = False
        
        # Load playlists for each emotion
        self._load_playlists()
        
        print("="*60)
        print("MUSIC PLAYER INITIALIZED")
        print("="*60)
        print(f"Music directory: {self.music_dir}")
        print(f"Volume: {int(volume * 100)}%")
        print(f"Transition delay: {transition_delay}s")
        print("\nLoaded playlists:")
        for emotion, folder in self.EMOTION_MAPPING.items():
            count = len(self.playlists.get(folder, []))
            print(f"  {emotion}: {count} tracks")
        print("="*60)
    
    def _load_playlists(self):
        """Load all music files from emotion-specific folders."""
        for emotion, folder_name in self.EMOTION_MAPPING.items():
            folder_path = self.music_dir / folder_name
            tracks = []
            
            if folder_path.exists() and folder_path.is_dir():
                # Find all audio files in the folder
                for file_path in folder_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.AUDIO_FORMATS:
                        tracks.append(str(file_path))
                
                # Shuffle for variety
                random.shuffle(tracks)
            
            self.playlists[folder_name] = tracks
    
    def _get_track_for_emotion(self, emotion: str, avoid_current: bool = False) -> Optional[str]:
        """
        Get a random track for the given emotion.
        
        Args:
            emotion: Emotion name (e.g., 'Happy', 'Sad')
            avoid_current: If True, avoid selecting the currently playing track
            
        Returns:
            Path to a music file, or None if no tracks available
        """
        folder_name = self.EMOTION_MAPPING.get(emotion)
        if not folder_name:
            return None
        
        playlist = self.playlists.get(folder_name, [])
        if not playlist:
            return None
        
        # If only one track, return it (can't avoid current)
        if len(playlist) == 1:
            return playlist[0]
        
        # Select random track, avoiding current if requested
        if avoid_current and self.current_track:
            available_tracks = [t for t in playlist if t != self.current_track]
            if available_tracks:
                return random.choice(available_tracks)
        
        # Select random track from playlist
        return random.choice(playlist)
    
    def _play_track(self, track_path: str):
        """
        Play a specific track.
        
        Args:
            track_path: Path to the music file
        """
        try:
            pygame.mixer.music.load(track_path)
            pygame.mixer.music.play()
            self.current_track = track_path
            self.is_playing = True
            self.is_paused = False
            print(f"Now playing: {Path(track_path).name}")
        except Exception as e:
            print(f"Error playing track {track_path}: {e}")
            self.is_playing = False
    
    def check_music_end(self):
        """
        Check if current song has ended and play next random song if auto-play is enabled.
        Should be called periodically (e.g., in main loop).
        Uses both event-based and polling methods for reliability.
        """
        if not self.auto_play_next or not self.current_emotion:
            return
        
        # Method 1: Check for music end event (requires video system)
        try:
            for event in pygame.event.get():
                if event.type == self.music_end_event:
                    # Song ended - play next random song from current emotion
                    if self.current_emotion and not self.is_paused:
                        print(f"Song ended. Playing next random {self.current_emotion} song...")
                        self._play_random_from_current_emotion()
                    return
        except pygame.error:
            # If event system fails, fall back to polling method
            pass
        
        # Method 2: Polling fallback - check if music stopped playing
        # This works even if video system isn't initialized
        if self.is_playing and not self.is_paused:
            if not pygame.mixer.music.get_busy():
                # Music was playing but now stopped - song ended
                if self.current_emotion:
                    print(f"Song ended. Playing next random {self.current_emotion} song...")
                    self._play_random_from_current_emotion()
    
    def _play_random_from_current_emotion(self):
        """
        Play a random song from the current emotion's playlist.
        Avoids playing the same song that just ended.
        """
        if not self.current_emotion:
            return
        
        # Get a random track, avoiding the one that just ended
        track = self._get_track_for_emotion(self.current_emotion, avoid_current=True)
        if track:
            # Don't fade out - just switch directly since song already ended
            self._play_track(track)
        else:
            print(f"No more tracks available for {self.current_emotion}")
            self.is_playing = False
    
    def _handle_emotion_transition(self, new_emotion: str):
        """
        Handle transition to a new emotion (with delay to avoid rapid switching).
        
        Args:
            new_emotion: New emotion detected
        """
        if new_emotion == self.current_emotion:
            return
        
        # Stop any existing transition
        self.stop_transition = True
        if self.transition_thread and self.transition_thread.is_alive():
            time.sleep(0.1)  # Give thread a moment to stop
        
        # Start new transition thread
        self.stop_transition = False
        self.transition_thread = threading.Thread(
            target=self._transition_worker,
            args=(new_emotion,),
            daemon=True
        )
        self.transition_thread.start()
    
    def _transition_worker(self, new_emotion: str):
        """
        Worker thread that waits for transition delay before switching music.
        
        Args:
            new_emotion: Emotion to transition to
        """
        # Wait for transition delay (checking for stop signal)
        elapsed = 0.0
        check_interval = 0.1
        
        while elapsed < self.transition_delay and not self.stop_transition:
            time.sleep(check_interval)
            elapsed += check_interval
        
        # If we weren't stopped, switch to new emotion
        if not self.stop_transition:
            self._switch_to_emotion(new_emotion)
    
    def _switch_to_emotion(self, emotion: str):
        """
        Switch to playing music for the given emotion.
        
        Args:
            emotion: Emotion name
        """
        track = self._get_track_for_emotion(emotion)
        
        if track:
            # Fade out current track if playing (but not if it's the first time)
            if self.is_playing and not self.is_paused and self.current_emotion is not None:
                # Only fade if we're actually switching emotions
                if self.current_emotion != emotion:
                    self.fade_out(duration=0.5)  # Shorter fade for faster switching
            elif not self.is_playing:
                # First time playing - start immediately
                pass
            
            # Play new track
            self._play_track(track)
            self.current_emotion = emotion
        else:
            print(f"No music available for emotion: {emotion}")
            self.current_emotion = emotion  # Still update emotion even if no music
    
    def update_emotion(self, emotion: str, confidence: float = 1.0, min_confidence: float = 0.5):
        """
        Update the current emotion and switch music if needed.
        
        Args:
            emotion: Detected emotion name
            confidence: Confidence score (0.0 to 1.0)
            min_confidence: Minimum confidence to switch music
        """
        if confidence < min_confidence:
            return  # Don't switch if confidence is too low
        
        if emotion != self.current_emotion:
            self._handle_emotion_transition(emotion)
    
    def play(self):
        """Resume playback if paused."""
        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.is_playing = True
            print("Music resumed")
        elif not self.is_playing and self.current_emotion:
            # Start playing if we have an emotion but nothing is playing
            self._switch_to_emotion(self.current_emotion)
    
    def pause(self):
        """Pause playback."""
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            print("Music paused")
    
    def stop(self):
        """Stop playback."""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.current_track = None
        print("Music stopped")
    
    def next_track(self):
        """Skip to next track in current emotion's playlist."""
        if self.current_emotion:
            self._switch_to_emotion(self.current_emotion)
    
    def switch_emotion_immediate(self, emotion: str):
        """
        Immediately switch to a new emotion's music (bypasses transition delay).
        Useful when emotion has been stable for a while.
        
        Args:
            emotion: Emotion name to switch to
        """
        if emotion != self.current_emotion:
            # Stop any pending transitions
            self.stop_transition = True
            if self.transition_thread and self.transition_thread.is_alive():
                time.sleep(0.1)
            
            # Switch immediately
            self._switch_to_emotion(emotion)
    
    def set_volume(self, volume: float):
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))  # Clamp to [0, 1]
        self.volume = volume
        pygame.mixer.music.set_volume(volume)
        print(f"Volume set to {int(volume * 100)}%")
    
    def increase_volume(self, step: float = 0.1):
        """Increase volume by step."""
        self.set_volume(self.volume + step)
    
    def decrease_volume(self, step: float = 0.1):
        """Decrease volume by step."""
        self.set_volume(self.volume - step)
    
    def fade_out(self, duration: float = 1.0):
        """
        Fade out current track.
        
        Args:
            duration: Fade duration in seconds
        """
        if not self.is_playing:
            return
        
        steps = 20
        step_dause = duration / steps
        initial_volume = self.volume
        
        for i in range(steps):
            new_volume = initial_volume * (1.0 - (i + 1) / steps)
            pygame.mixer.music.set_volume(new_volume)
            time.sleep(step_dause)
        
        pygame.mixer.music.stop()
        pygame.mixer.music.set_volume(initial_volume)  # Restore volume
        self.is_playing = False
    
    def get_current_track_name(self) -> str:
        """Get the name of the currently playing track."""
        if self.current_track:
            return Path(self.current_track).stem
        return "No track"
    
    def get_status(self) -> Dict:
        """Get current player status."""
        return {
            'emotion': self.current_emotion,
            'track': self.get_current_track_name(),
            'playing': self.is_playing,
            'paused': self.is_paused,
            'volume': int(self.volume * 100)
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        pygame.mixer.quit()


if __name__ == "__main__":
    # Test the music player with actual playback
    print("="*60)
    print("MUSIC PLAYER TEST - Playing Music Out Loud")
    print("="*60)
    
    # Check if music directory exists
    music_dir = Path('music')
    if not music_dir.exists():
        print(f"\nWarning: Music directory '{music_dir}' not found.")
        print("Creating directory structure...")
        for emotion_folder in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            (music_dir / emotion_folder).mkdir(parents=True, exist_ok=True)
        print(f"Created music directory structure at '{music_dir}'")
        print("Please add music files to the emotion-specific folders.")
    else:
        # Initialize player with shorter transition delay for demo
        player = EmotionMusicPlayer(music_dir='music', volume=0.7, transition_delay=2.0)
        
        # Check if we have any music files
        total_tracks = sum(len(playlist) for playlist in player.playlists.values())
        if total_tracks == 0:
            print("\n⚠️  No music files found in emotion folders!")
            print("Please add music files to the emotion-specific folders:")
            for emotion, folder in player.EMOTION_MAPPING.items():
                print(f"  - music/{folder}/")
            print("\nSupported formats: MP3, WAV, OGG, M4A, FLAC")
            player.cleanup()
            exit(1)
        
        print(f"\n✓ Found {total_tracks} total music tracks")
        print("\n" + "="*60)
        print("Starting music playback demo...")
        print("Press Ctrl+C to stop early")
        print("="*60)
        
        # Test emotion switching with actual playback
        emotions = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Fear', 'Disgust']
        
        try:
            # Start with first emotion immediately (no transition delay)
            first_emotion = emotions[0]
            print(f"\n[1/{len(emotions)}] Starting with {first_emotion}...")
            player._switch_to_emotion(first_emotion)  # Direct switch for first emotion
            if player.is_playing:
                current_track = player.get_current_track_name()
                print(f"✓ Now playing: {current_track}")
                print(f"  Volume: {int(player.volume * 100)}%")
                play_duration = 10
                print(f"  Playing for {play_duration} seconds...")
                time.sleep(play_duration)
            else:
                print(f"⚠️  No music available for {first_emotion}")
            
            # Continue with remaining emotions
            for i, emotion in enumerate(emotions[1:], 2):
                print(f"\n[{i}/{len(emotions)}] Switching to {emotion}...")
                
                # Update emotion (this will trigger music after transition delay)
                player.update_emotion(emotion, confidence=0.9)
                
                # Wait for transition delay + play time
                print(f"Waiting for transition ({player.transition_delay}s)...")
                time.sleep(player.transition_delay + 0.5)  # Wait for transition
                
                # Check if music is playing
                if player.is_playing:
                    current_track = player.get_current_track_name()
                    print(f"✓ Now playing: {current_track}")
                    print(f"  Volume: {int(player.volume * 100)}%")
                    
                    # Play for 8-10 seconds so user can hear it
                    play_duration = 10
                    print(f"  Playing for {play_duration} seconds...")
                    time.sleep(play_duration)
                else:
                    print(f"⚠️  No music available for {emotion}")
                    time.sleep(2)
            
            print("\n" + "="*60)
            print("Demo complete! Playing all emotions in sequence...")
            print("="*60)
            
            # Final demo: cycle through all emotions quickly
            print("\nQuick cycle through all emotions (5 seconds each)...")
            for emotion in emotions:
                print(f"\n→ {emotion}")
                player.update_emotion(emotion, confidence=0.9)
                time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            print("\n" + "="*60)
            print("Stopping music player...")
            player.cleanup()
            print("✓ Music player stopped. Demo complete!")
            print("="*60)

