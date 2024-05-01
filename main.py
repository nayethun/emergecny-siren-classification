import numpy as np

def buzz_alert(audio_segments):
    """
    Function to determine general alert for BUzz bracelet.
    
    Input:
        audio_segments (numpy.ndarray): 3xN matrix where each column represents 1-second long segments of audio.
    
    Returns:
        Alert (int): 1 if the alert condition is met, 0 otherwise.
        audio_segments (numpy.ndarray): Updated audio segments with the oldest segment removed.
    """
    # Calculate RMS values for each segment
    audio_rms = np.zeros(3)
    audio_rms[0] = np.sqrt(np.mean(np.square(audio_segments[:, 0])))
    audio_rms[1] = np.sqrt(np.mean(np.square(audio_segments[:, 1])))
    audio_rms[2] = np.sqrt(np.mean(np.square(audio_segments[:, 2])))
    
    # Threshold for alert
    thresh = 1.225
    # Calculate threshold for current audio using avg of first 2 seconds
    audio_thresh = (audio_rms[0] + audio_rms[1]) / 2 * thresh
    
    # Compare threshold to most recent second of audio
    if audio_thresh < audio_rms[2]:
        alert = 1
    else:
        alert = 0

    # Shift input matrix to make space for new second of audio
    audio_segments[:, 0:2] = audio_segments[:, 1:3]
    audio_segments[:, 2] = 0  # Reset the last segment, ready for new data
    
    return alert, audio_segments
