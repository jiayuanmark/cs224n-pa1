package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;

public class PMIModel implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;

	// target conditioned by source
	private CounterMap<String, String> conditionalCounter;
	
	public PMIModel() {
		conditionalCounter = new CounterMap<String, String>();
	}
	
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment align = new Alignment();
		int numSrcWords = sentencePair.getSourceWords().size();
		int numTgtWords = sentencePair.getTargetWords().size();
		List<String> srcWords = sentencePair.getSourceWords();
		List<String> tgtWords = sentencePair.getTargetWords();
		
		for (int tgtIdx = 0; tgtIdx < numTgtWords; ++tgtIdx) {
			// Initialize with a null alignment
			double score = conditionalCounter.getCount(NULL_WORD, tgtWords.get(tgtIdx));
			int maxIdx = -1;
			
			for (int srcIdx = 0; srcIdx < numSrcWords; ++srcIdx) {
				if (conditionalCounter.getCount(srcWords.get(srcIdx), tgtWords.get(tgtIdx)) > score) {
					score = conditionalCounter.getCount(srcWords.get(srcIdx), tgtWords.get(tgtIdx));
					maxIdx = srcIdx;
				}
			}
			
			// Drop null alignment
			if (maxIdx != -1)
				align.addPredictedAlignment(tgtIdx, maxIdx);
		}
		return align;
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		// For each of the sentence pair
		for (SentencePair pair : trainingData) {
			
			// count(f_j, e_i) where j = 1, ..., m 
			for (String srcWord : pair.getSourceWords())
				for (String tgtWord : pair.getTargetWords())
					conditionalCounter.incrementCount(srcWord, tgtWord, 1.0);
			
			// count(f_0, e_i)
			for (String tgtWord : pair.getTargetWords())
				conditionalCounter.incrementCount(NULL_WORD, tgtWord, 1.0);
		}
		
		// Normalize to get p(e_i | f_j)
		conditionalCounter = Counters.conditionalNormalize(conditionalCounter);
	}

}
