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

	private void initialize(List<SentencePair> trainingData) {
		for (SentencePair pair : trainingData) {
			for (String srcWord : pair.getSourceWords())
                                for (String tgtWord : pair.getTargetWords())
                                        conditionalCounter.incrementCount(srcWord, tgtWord, 1.0);
			for (String tgtWord : pair.getTargetWords())
                                conditionalCounter.incrementCount(NULL_WORD, tgtWord, 1.0);
		}
		conditionalCounter = Counters.conditionalNormalize(conditionalCounter);
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		// For each of the sentence pair
		initialize(trainingData);
		CounterMap<String, String> currentConditionalCounter;
		while (true) {

			currentConditionalCounter = new CounterMap<String, String>();
			for (SentencePair pair : trainingData) {
				// count(f_j, e_i) where j = 1, ..., m 
				for (String tgtWord : pair.getTargetWords()) {
					double sum = 0.0;
					for (String srcWord : pair.getSourceWords())
						sum += conditionalCounter.getCount(srcWord, tgtWord);
					sum += conditionalCounter.getCount(NULL_WORD, tgtWord);

					for (String srcWord : pair.getSourceWords())
						currentConditionalCounter.incrementCount(srcWord, tgtWord, conditionalCounter.getCount(srcWord, tgtWord)/sum);
			
				// count(f_0, e_i)
					currentConditionalCounter.incrementCount(NULL_WORD, tgtWord, conditionalCounter.getCount(NULL_WORD, tgtWord)/sum);
				}
			}
		
			// Normalize to get p(e_i | f_j)	
			currentConditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			if (conditionalCounter.compareCounter(currentConditionalCounter) < 0.001)
				break;
			conditionalCounter = currentConditionalCounter;
		}
	}

}
