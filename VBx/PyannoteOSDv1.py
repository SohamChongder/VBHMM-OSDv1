from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
import os

def postprocessing_osd_single_file(audio_path, rttm_dir, overlap_dir, rttm_osd_dir):
    # Initialize the pyannote audio pipeline for OSD
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
    pipeline = OverlappedSpeechDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    # Construct the corresponding RTTM paths
    rttm_path = os.path.join(rttm_dir, os.path.basename(audio_path).replace('.wav', '.rttm'))
    output_rttm_path = os.path.join(rttm_osd_dir, os.path.basename(audio_path).replace('.wav', '.rttm'))
    overlap_rttm_path = os.path.join(overlap_dir, os.path.basename(audio_path).replace('.wav', '.rttm'))
    
    # Load the original RTTM file into an Annotation
    diarization = load_rttm(rttm_path)
    if len(diarization) != 1:
        raise ValueError("RTTM file must contain exactly one recording.")
    diarization = list(diarization.values())[0]
    
    # Detect overlaps using pyannote.audio
    osd = pipeline(audio_path)
    overlaps = osd.get_timeline().support()
    
    os.makedirs(overlap_dir, exist_ok=True)
    os.makedirs(rttm_osd_dir, exist_ok=True)
    
    # Save the overlaps as a separate RTTM file
    overlap_annotation = Annotation()
    for overlap in overlaps:
        overlap_annotation[overlap, 'overlap'] = 'overlap'
    with open(overlap_rttm_path, 'w') as f:
        f.write(overlap_annotation.to_rttm())
    print(f"Overlapped regions saved to: {overlap_rttm_path}")
    
    # Create updated RTTM with multiple speakers in overlaps
    updated_diarization = diarization.copy()
    for overlap in overlaps:
        # Get speakers in the overlap
        overlap_speakers = diarization.crop(overlap).labels()

        if len(overlap_speakers) == 1:
            primary_speaker = overlap_speakers[0]
            
            # Search before the overlap
            speakers_before = diarization.crop(Segment(0, overlap.start)).labels()
            speakers_before = [s for s in speakers_before if s != primary_speaker]
            closest_before = speakers_before[-1] if speakers_before else None

            # Search after the overlap
            speakers_after = diarization.crop(Segment(overlap.end, diarization.get_timeline().extent().end)).labels()
            speakers_after = [s for s in speakers_after if s != primary_speaker]
            closest_after = speakers_after[0] if speakers_after else None

            # Determine the closest different speaker
            closest_speaker = closest_before or closest_after

            # Add both speakers to the RTTM for the overlap
            if closest_speaker:
                updated_diarization[Segment(overlap.start, overlap.end), closest_speaker] = closest_speaker
    
    # Save updated RTTM to output file
    with open(output_rttm_path, 'w') as f:
        f.write(updated_diarization.to_rttm())
    print(f"Postprocessed RTTM saved to: {output_rttm_path}")

def postprocess_diarized_rttm(wav_scp_file):
    # Read the wav.scp file to get all audio paths
    with open(wav_scp_file, 'r') as f:
        audio_paths = [line.strip().split()[1] for line in f.readlines()]
    
    # Loop through each audio path and process
    for audio_path in audio_paths:
        postprocessing_osd_single_file(audio_path, "exp/rttm", "exp/overlap", "exp/rttm_osd")

# Example usage (requires an 'args' object with appropriate attributes)
# postprocess_diarized_rttm("path/to/wav.scp", args)
