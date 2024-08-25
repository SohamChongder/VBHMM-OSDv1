path_segments=dev_final_seg
path_new_kaldi_segs=dev_final_seg_kaldi
for path in $(ls $path_segments)
	do

			g=$path_segments/$path
			python3 SEG_TO_KALDI_SEG.py $g $path_new_kaldi_segs/$path.txt
		
	done
	


