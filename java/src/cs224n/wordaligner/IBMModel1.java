package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;

public class IBMModel1 implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double min_delta = 0.00001;

	// target conditioned by source
	private CounterMap<String, String> conditionalCounter;
	
	public IBMModel1() {
		conditionalCounter = new CounterMap<String, String>();
	}
	
	public CounterMap<String, String> getConditionalCounter() {
		return conditionalCounter;
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
			double score = conditionalCounter.getCount(tgtWords.get(tgtIdx), NULL_WORD);
			int maxIdx = -1;
			
			for (int srcIdx = 0; srcIdx < numSrcWords; ++srcIdx) {
				if (conditionalCounter.getCount(tgtWords.get(tgtIdx), srcWords.get(srcIdx)) > score) {
					score = conditionalCounter.getCount(tgtWords.get(tgtIdx), srcWords.get(srcIdx));
					maxIdx = srcIdx;
				}
			}
			
			// Drop null alignment
			if (maxIdx != -1)
				align.addPredictedAlignment(tgtIdx, maxIdx);
		}
		return align;
	}

	private void initialize(List<SentencePair> trainingData) {
		for (SentencePair pair : trainingData) {
			for (String srcWord : pair.getSourceWords())
                                for (String tgtWord : pair.getTargetWords())
					if (conditionalCounter.getCount(tgtWord, srcWord) == 0)
                                        	conditionalCounter.incrementCount(tgtWord, srcWord, 1.0);
			for (String tgtWord : pair.getTargetWords())
				if (conditionalCounter.getCount(tgtWord, NULL_WORD) == 0)
                                	conditionalCounter.incrementCount(tgtWord, NULL_WORD, 1.0);
		}
		conditionalCounter = Counters.conditionalNormalize(conditionalCounter);
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		// For each of the sentence pair
		initialize(trainingData);
		int i = 0;
		CounterMap<String, String> currentConditionalCounter;
		while (i < 1000) {

			currentConditionalCounter = new CounterMap<String, String>();
			for (SentencePair pair : trainingData) {
				// count(f_j, e_i) where j = 1, ..., m 
				for (String srcWord : pair.getSourceWords())
					for (String tgtWord : pair.getTargetWords())
						currentConditionalCounter.incrementCount(tgtWord, srcWord, conditionalCounter.getCount(tgtWord, srcWord));
			
				// count(f_0, e_i)
				for (String tgtWord : pair.getTargetWords())
					conditionalCounter.incrementCount(tgtWord, NULL_WORD, conditionalCounter.getCount(tgtWord, NULL_WORD));
			}
		
			// Normalize to get p(e_i | f_j)	
			conditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			i ++;
		}
	}

}
